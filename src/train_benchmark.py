import argparse
import codecs
import fnmatch
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# Written into checkpoint-<step>/ only after every rank finished saving it.
CHECKPOINT_COMPLETE_MARKER = "_COMPLETE"
# Written into milestone checkpoints (milestone_percents); never rotated.
MILESTONE_FILE = "MILESTONE.json"


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def resolve_output_base(cfg) -> Path:
    """output_dir, placed under checkpoint_root when it is relative."""
    out = Path(cfg["output_dir"])
    if cfg.get("checkpoint_root") and not out.is_absolute():
        out = Path(cfg["checkpoint_root"]) / out
    return resolve_path(str(out))


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict):
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_model_config(config_path: Path, model_key: str, qlora: bool = False):
    """
    Build the effective config for one model from the central config.

    BF16 : defaults < models.<key>
    QLoRA: defaults < models.<key> < qlora < models.<key>.qlora
    """
    root = load_yaml(config_path)
    models = root.get("models", {})

    if model_key not in models:
        raise KeyError(
            f"Unknown model '{model_key}'. "
            f"Available: {', '.join(models)}"
        )

    model_cfg = dict(models[model_key])
    model_qlora_cfg = model_cfg.pop("qlora", None) or {}

    cfg = deep_merge(root.get("defaults", {}), model_cfg)
    if qlora:
        cfg = deep_merge(cfg, root.get("qlora", {}))
        cfg = deep_merge(cfg, model_qlora_cfg)

    cfg["name"] = model_key
    return cfg


def path_matches_exclude(rel_parts, pattern_parts) -> bool:
    """True if pattern_parts (fnmatch per component) appear as consecutive
    components anywhere in rel_parts, i.e. the file is that path or below it."""
    n = len(pattern_parts)
    for off in range(len(rel_parts) - n + 1):
        if all(fnmatch.fnmatchcase(rel_parts[off + i], pattern_parts[i]) for i in range(n)):
            return True
    return False


def discover_documents(input_path: Path, pattern: str, exclude_name_keywords=(), exclude_paths=()):
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        # File names may be NFD (macOS origin), so compare in NFC.
        keywords = [unicodedata.normalize("NFC", k) for k in exclude_name_keywords]
        path_patterns = [
            [c for c in unicodedata.normalize("NFC", e).replace("\\", "/").split("/") if c]
            for e in exclude_paths
        ]
        files, by_name, by_path = [], [], []
        for p in sorted(input_path.glob(pattern)):
            if not p.is_file():
                continue
            name = unicodedata.normalize("NFC", p.name)
            rel_parts = unicodedata.normalize("NFC", p.relative_to(input_path).as_posix()).split("/")
            if any(k in name for k in keywords):
                by_name.append(p)
            elif any(path_matches_exclude(rel_parts, pp) for pp in path_patterns):
                by_path.append(p)
            else:
                files.append(p)
        for excluded, why in ((by_name, f"name keyword {keywords}"), (by_path, "exclude_paths")):
            if excluded:
                print(f"[DATA] Excluded {len(excluded)} file(s) by {why}:")
                for p in excluded:
                    print(f"[DATA]   - {unicodedata.normalize('NFC', str(p.relative_to(input_path)))}")
        return files

    raise FileNotFoundError(f"Data path not found: {input_path}")


def dataset_signature(files, model_path: Path, seq_len: int, add_eos: bool,
                      separator=None, shuffle: bool = False, data_seed: int = 0,
                      end_token=None):
    """
    Lightweight signature used only to detect obvious cache mismatches.
    It hashes paths, file sizes, mtimes, tokenizer/model path, and key settings.
    It does not hash the entire multi-GB corpus.
    """
    h = hashlib.sha256()
    h.update(str(model_path).encode())
    h.update(str(seq_len).encode())
    h.update(str(add_eos).encode())
    h.update(repr((separator, shuffle, data_seed, end_token)).encode())

    for p in sorted(files):
        stat = p.stat()
        h.update(str(p.resolve()).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())

    return h.hexdigest()


def find_documents(file_path: Path, separator, block_bytes: int = 64 << 20):
    """
    Byte spans (start, end) of the documents in a file, split at the literal
    `separator` (which is left out). Without a separator the whole file is
    one document. Empty spans are skipped.
    """
    size = file_path.stat().st_size
    if not separator:
        return [(0, size)] if size else []

    sep = separator.encode("utf-8")
    spans, doc_start, pos, tail = [], 0, 0, b""
    with file_path.open("rb") as f:
        while True:
            block = f.read(block_bytes)
            if not block:
                break
            data = tail + block
            base = pos - len(tail)
            i = 0
            while True:
                j = data.find(sep, i)
                if j < 0:
                    break
                spans.append((doc_start, base + j))
                doc_start = base + j + len(sep)
                i = j + len(sep)
            # Keep enough bytes to catch a separator split across blocks.
            tail = data[max(i, len(data) - (len(sep) - 1)):]
            pos += len(block)
    spans.append((doc_start, size))
    return [(a, b) for a, b in spans if b > a]


def iter_span_chunks(file_path: Path, start: int, end: int, chunk_chars: int):
    """
    Stream bytes [start, end) of a file as text pieces of about chunk_chars
    bytes, cut at a line break where possible (never inside a character).

    Piece boundaries are not document boundaries and do not get EOS.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    with file_path.open("rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            block = f.read(min(chunk_chars, remaining))
            if not block:
                break
            remaining -= len(block)
            pending += decoder.decode(block, final=remaining == 0)
            if remaining == 0:
                break
            cut = pending.rfind("\n") + 1
            if cut == 0:
                if len(pending) < chunk_chars:
                    continue
                cut = len(pending)  # one very long line
            yield pending[:cut]
            pending = pending[cut:]
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending


def packed_generator(
    documents,
    tokenizer,
    seq_len,
    add_eos_between_documents,
    chunk_chars,
    eos_id=None,
):
    token_buffer = []
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    for file_path, start, end in documents:
        file_path = Path(file_path)
        for text_chunk in iter_span_chunks(file_path, start, end, chunk_chars):
            if not text_chunk:
                continue

            ids = tokenizer(
                text_chunk,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            token_buffer.extend(ids)

            while len(token_buffer) >= seq_len:
                yield {"input_ids": token_buffer[:seq_len]}
                del token_buffer[:seq_len]

        if add_eos_between_documents and eos_id is not None:
            token_buffer.append(eos_id)

            while len(token_buffer) >= seq_len:
                yield {"input_ids": token_buffer[:seq_len]}
                del token_buffer[:seq_len]

    # Intentionally drop the final short remainder.
    # Every cached sequence therefore has exactly seq_len tokens.


def cache_ready(cache_path: Path):
    return cache_path.exists() and (cache_path / "_READY").exists()


def read_cache_metadata(cache_path: Path):
    metadata_path = cache_path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_cache_if_needed(cfg, rank: int, rebuild_cache: bool):
    model_path = resolve_path(cfg["model_path"])
    data_path = resolve_path(cfg["data_path"])
    cache_path = resolve_path(cfg["cache_path"])

    seq_len = int(cfg.get("seq_len", 2048))
    pattern = cfg.get("file_pattern", "**/*.txt")
    shuffle_documents = bool(cfg.get("shuffle_documents", True))
    data_seed = int(cfg.get("data_seed", 0))
    separator = cfg.get("document_separator") or None
    end_token = cfg.get("document_end_token") or None
    add_eos = bool(cfg.get("add_eos_between_documents", True))
    chunk_chars = int(cfg.get("tokenize_chunk_chars", 1_000_000))

    exclude_name_keywords = cfg.get("exclude_name_keywords") or []
    exclude_paths = cfg.get("exclude_paths") or []
    files = discover_documents(data_path, pattern, exclude_name_keywords, exclude_paths)
    if not files:
        raise RuntimeError(f"No .txt files found at: {data_path}")

    signature = dataset_signature(
        files, model_path, seq_len, add_eos, separator, shuffle_documents, data_seed,
        end_token,
    )

    # Rank 0 decides whether the cache is valid and creates it.
    if rank == 0:
        if cache_ready(cache_path) and not rebuild_cache:
            metadata = read_cache_metadata(cache_path)
            if metadata.get("signature") == signature:
                print(f"[CACHE] Reusing existing cache: {cache_path}")
                return metadata

            print("[CACHE] Existing cache does not match the current data/config.")
            print("[CACHE] Rebuilding it.")

        if cache_path.exists():
            shutil.rmtree(cache_path)

        build_path = cache_path.with_name(cache_path.name + ".building")
        if build_path.exists():
            shutil.rmtree(build_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        documents = [
            (str(p), start, end)
            for p in files
            for start, end in find_documents(p, separator)
        ]
        if shuffle_documents:
            rng = random.Random(data_seed)
            rng.shuffle(documents)

        print("\n" + "=" * 80)
        print("TOKENIZATION / PACKING (RANK 0 ONLY)")
        print("=" * 80)
        print(f"Tokenizer model : {model_path}")
        print(f"Raw data        : {data_path}")
        print(f"Files           : {len(files):,}")
        print(f"Documents       : {len(documents):,} (separator {separator!r})")
        print(f"Shuffle / seed  : {shuffle_documents} / {data_seed}")
        print(f"Seq length      : {seq_len}")
        print(f"EOS per document: {add_eos}")
        print(f"Cache           : {cache_path}")
        print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True,
        )

        if end_token:
            eos_id = tokenizer.convert_tokens_to_ids(end_token)
            if eos_id is None or eos_id == tokenizer.unk_token_id:
                raise ValueError(f"document_end_token {end_token!r} is not a token of {model_path}")
        else:
            eos_id = tokenizer.eos_token_id
        print(f"Document end    : {tokenizer.convert_ids_to_tokens(eos_id)!r} (id {eos_id})")

        start = time.perf_counter()

        dataset = Dataset.from_generator(
            packed_generator,
            gen_kwargs={
                # A tuple, not a list: Dataset.from_generator splits list
                # kwargs into shards and calls the generator once per item,
                # which would drop every document's last partial sequence.
                "documents": tuple(documents),
                "tokenizer": tokenizer,
                "seq_len": seq_len,
                "add_eos_between_documents": add_eos,
                "chunk_chars": chunk_chars,
                "eos_id": eos_id,
            },
        )

        dataset.save_to_disk(str(build_path))

        elapsed = time.perf_counter() - start
        num_sequences = len(dataset)
        total_tokens = num_sequences * seq_len

        metadata = {
            "signature": signature,
            "model_name": cfg["name"],
            "model_path": str(model_path),
            "raw_data_path": str(data_path),
            "num_files": len(files),
            "num_documents": len(documents),
            "seq_len": seq_len,
            "num_sequences": num_sequences,
            "total_tokens": total_tokens,
            "preprocessing_seconds": elapsed,
        }

        with (build_path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        (build_path / "_READY").write_text("ready\n", encoding="utf-8")
        build_path.rename(cache_path)

        print("\n" + "=" * 80)
        print("CACHE READY")
        print("=" * 80)
        print(f"Sequences      : {num_sequences:,}")
        print(f"Tokens         : {total_tokens:,}")
        print(f"Preprocess time: {elapsed:.2f} sec")
        print(f"Cache          : {cache_path}")
        print("=" * 80)

        return metadata

    # Other ranks wait for rank 0 instead of repeating tokenization.
    print(f"[RANK {rank}] Waiting for rank 0 to prepare cache...")

    timeout_seconds = int(cfg.get("cache_wait_timeout_seconds", 86400))
    start_wait = time.time()

    while not cache_ready(cache_path):
        if time.time() - start_wait > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for cache: {cache_path}"
            )
        time.sleep(5)

    return read_cache_metadata(cache_path)


class CausalLMCollator:
    def __call__(self, features):
        input_ids = torch.tensor(
            [item["input_ids"] for item in features],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def list_checkpoints(run_dir: Path):
    """checkpoint-<step> directories in run_dir, oldest first."""
    checkpoints = []
    for p in run_dir.glob("checkpoint-*"):
        suffix = p.name.split("-", 1)[1]
        if p.is_dir() and suffix.isdigit():
            checkpoints.append((int(suffix), p))
    return [p for _, p in sorted(checkpoints)]


def find_resume_checkpoint(run_dir: Path, rank: int):
    """
    Return the newest checkpoint that finished saving, or None.

    A checkpoint without the marker was cut off mid-save (e.g. the GPU
    reservation ended during the save). Rank 0 deletes those so they are
    not resumed from and do not confuse checkpoint rotation.
    """
    latest = None
    for checkpoint in list_checkpoints(run_dir):
        if (checkpoint / CHECKPOINT_COMPLETE_MARKER).exists():
            latest = checkpoint
        elif rank == 0:
            print(f"[CKPT] Removing incomplete checkpoint: {checkpoint}")
            shutil.rmtree(checkpoint, ignore_errors=True)
    return latest


DATA_STATE_FILE = "data_state.json"

# Settings that decide which batch comes next. Trainer resumes the data by
# replaying the same seeded shuffle and skipping the batches already used,
# so the data only continues where it stopped if all of these are unchanged.
DATA_ORDER_KEYS = (
    "dataset_signature",
    "num_sequences",
    "shuffle_documents",
    "data_seed",
    "seed",
    "world_size",
    "micro_batch_size",
    "gradient_accumulation_steps",
)


def check_data_state(checkpoint: Path, expected: dict, rank: int):
    """Refuse to resume if the data order or batch layout has changed."""
    path = checkpoint / DATA_STATE_FILE
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing, so it cannot be verified that the data "
            f"continues where it stopped. Use --no_resume to start over."
        )

    with path.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    changed = [
        f"  {key}: checkpoint={saved.get(key)!r} now={expected[key]!r}"
        for key in DATA_ORDER_KEYS
        if saved.get(key) != expected[key]
    ]
    if changed:
        raise RuntimeError(
            "Cannot resume: these settings changed since the checkpoint, so "
            "the data would not continue where it stopped:\n"
            + "\n".join(changed)
            + "\nRestore them, or start over with --no_resume."
        )

    if rank == 0:
        print(
            f"[CKPT] Data resumes after {saved['consumed_sequences']:,} sequences "
            f"({saved['consumed_tokens']:,} tokens), epoch {saved['epoch']:.4f}"
        )


class CheckpointCompleteMarker(TrainerCallback):
    """
    Finish a checkpoint once every rank has written its shard:
    record the data position, mark it complete, then delete older
    checkpoints so only the newest `keep` remain.

    Rotation is done here instead of by Trainer (save_total_limit) because
    Trainer rotates on rank 0 as soon as rank 0 itself is done; with one
    kept checkpoint it could delete the last good one while other ranks are
    still writing the new one. The new checkpoint is never written over
    the old one in place, so a shutdown mid-save always leaves one intact.
    """

    def __init__(self, keep: int, data_state: dict, milestones=None,
                 weights_only: bool = False, tokenizer_path: Path = None):
        self.keep = keep
        self.data_state = data_state
        self.milestones = milestones
        self.weights_only = weights_only
        self.tokenizer_path = tokenizer_path
        self.trainer = None  # set once the Trainer exists (weights export)

    def on_save(self, args, state, control, **kwargs):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        percent = self.milestones.steps.get(state.global_step) if self.milestones else None
        if percent is not None and self.weights_only:
            # Before _COMPLETE: if the export is cut off, this checkpoint is
            # incomplete too, so the resume replays this step and exports again.
            export_milestone_weights(
                self.trainer, Path(args.output_dir), percent, self.tokenizer_path
            )

        if not state.is_world_process_zero:
            return

        run_dir = Path(args.output_dir)
        checkpoint = run_dir / f"checkpoint-{state.global_step}"
        if not checkpoint.is_dir():
            return

        sequences_per_step = (
            self.data_state["world_size"]
            * self.data_state["micro_batch_size"]
            * self.data_state["gradient_accumulation_steps"]
        )
        consumed = state.global_step * sequences_per_step
        data_state = {
            **self.data_state,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "consumed_sequences": consumed,
            "consumed_tokens": consumed * self.data_state["seq_len"],
        }
        with (checkpoint / DATA_STATE_FILE).open("w", encoding="utf-8") as f:
            json.dump(data_state, f, indent=2)

        if percent is not None:
            with (checkpoint / MILESTONE_FILE).open("w", encoding="utf-8") as f:
                json.dump(
                    {"percent": percent, "step": state.global_step,
                     "max_steps": state.max_steps, "epoch": state.epoch},
                    f, indent=2,
                )

        (checkpoint / CHECKPOINT_COMPLETE_MARKER).write_text(
            "complete\n", encoding="utf-8"
        )
        if percent is not None and not self.weights_only:
            print(f"[CKPT] Saved milestone {percent}%: {checkpoint}")
        else:
            print(f"[CKPT] Saved: {checkpoint}")

        if self.keep > 0:
            # Full milestone checkpoints are kept; resume checkpoints (and,
            # with weights-only milestones, every checkpoint-<step>) rotate.
            others = [
                c for c in list_checkpoints(run_dir)
                if c != checkpoint
                and (self.weights_only or milestone_percent(c) is None)
            ]
            for old in others[: max(0, len(others) - (self.keep - 1))]:
                print(f"[CKPT] Removing old checkpoint: {old}")
                shutil.rmtree(old, ignore_errors=True)


def milestone_percent(checkpoint: Path):
    """Percent recorded in a complete milestone checkpoint, else None."""
    path = checkpoint / MILESTONE_FILE
    if not (path.exists() and (checkpoint / CHECKPOINT_COMPLETE_MARKER).exists()):
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("percent")


class MilestoneSaver(TrainerCallback):
    """
    Save at each of milestone_percents of max_steps. With weights_only,
    milestones before the last step export only the bf16 weights; the last
    step also saves the full training state (checkpoint-<step>/).
    """

    def __init__(self, percents, weights_only: bool = False, tokenizer_path: Path = None):
        self.percents = sorted({float(p) for p in percents or []})
        self.steps = {}
        self.weights_only = weights_only
        self.tokenizer_path = tokenizer_path
        self.trainer = None  # set once the Trainer exists (weights export)

    def on_train_begin(self, args, state, control, **kwargs):
        # state.max_steps is final here (also when resuming).
        self.steps = {}
        for p in self.percents:
            step = max(1, math.ceil(state.max_steps * p / 100))
            self.steps[step] = int(p) if p.is_integer() else p
        if state.is_world_process_zero and self.steps:
            plan = ", ".join(f"{p}% @ step {s}" for s, p in sorted(self.steps.items()))
            print(f"[CKPT] Milestones: {plan}")

    def on_step_end(self, args, state, control, **kwargs):
        percent = self.steps.get(state.global_step)
        if percent is None:
            return control
        if self.weights_only and state.global_step < state.max_steps:
            export_milestone_weights(
                self.trainer, Path(args.output_dir), percent, self.tokenizer_path
            )
        else:
            control.should_save = True
        return control


class CheckpointRootBusy(RuntimeError):
    pass


def check_checkpoint_root(cfg, run_dir: Path):
    """
    Refuse to start when this run already finished (100% milestone), so a
    finished run is never trained over. Other models' runs, running or
    finished, do not block. Never deletes anything: clearing is done by
    hand with scripts/clear_checkpoints.py.
    """
    if not cfg.get("checkpoint_root"):
        return
    root = resolve_path(cfg["checkpoint_root"])
    if not root.is_dir():
        return

    candidates = [
        p for p in root.rglob("checkpoint-*")
        if p.is_dir() and p.name.split("-", 1)[1].isdigit()
    ]
    candidates += [p for p in root.rglob("milestone-*") if p.is_dir()]

    finished = [
        c for c in sorted(candidates)
        if c.parent == run_dir and milestone_percent(c) == 100
    ]

    hint = (
        "Copy the milestone checkpoints off this machine, then clear them by "
        "hand with:\n  python scripts/clear_checkpoints.py"
    )
    if finished:
        raise CheckpointRootBusy(
            f"Not starting: this run already finished (100% milestone at "
            f"{finished[-1]}).\n{hint}"
        )


def run_dir_for(cfg, args, run_name_prefix: str, world_size: int) -> Path:
    attn_impl = (
        args.attn_implementation
        or cfg.get("attn_implementation", "flash_attention_2")
    )
    return resolve_output_base(cfg) / f"{run_name_prefix}{attn_impl}_{world_size}gpu"


def preflight(cfg, args, run_name_prefix: str = ""):
    """--preflight: run the checkpoint_root check without training, then exit."""
    if not args.preflight:
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        check_checkpoint_root(cfg, run_dir_for(cfg, args, run_name_prefix, world_size))
    except CheckpointRootBusy as e:
        print(f"[PREFLIGHT] {e}")
        sys.exit(2)
    print("[PREFLIGHT] OK")
    sys.exit(0)


def distributed_any(flags):
    """Element-wise OR of boolean flags across ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return [bool(f) for f in flags]

    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else "cpu"
    )
    tensor = torch.tensor(
        [1 if f else 0 for f in flags],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [bool(v) for v in tensor.tolist()]


class InterruptionGuard(TrainerCallback):
    """
    Extra saves for runs on a reserved (time-limited) GPU:

      - save every `save_interval_minutes` (in addition to save_steps),
      - save and stop cleanly `stop_margin_minutes` before `stop_at`,
      - save and stop cleanly on SIGTERM / SIGINT.

    Decisions are all-reduced so every rank saves/stops at the same step.
    Note: torchrun kills its workers ~30 s after torchrun itself is
    signalled, so stop through scripts/run_benchmark.sh (it signals the
    workers directly) or rely on stop_at for large checkpoints.
    """

    def __init__(self, save_interval_minutes: float, stop_at, stop_margin_minutes: float):
        self.save_interval = save_interval_minutes * 60
        self.deadline = (
            stop_at.timestamp() - stop_margin_minutes * 60
            if stop_at is not None
            else None
        )
        self.last_save = time.time()
        self.signal_name = None
        self.stop_reason = None

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, frame):
        if self.signal_name is not None:
            # Ctrl+C / group kills reach a worker more than once; keep saving.
            # Use SIGKILL to abort without saving.
            return
        self.signal_name = signal.Signals(signum).name
        print(
            f"[GUARD] {self.signal_name} received: "
            "saving a checkpoint after the current step, then stopping."
        )

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        got_signal, past_deadline, interval_due = distributed_any([
            self.signal_name is not None,
            self.deadline is not None and now >= self.deadline,
            self.save_interval > 0 and now - self.last_save >= self.save_interval,
        ])

        if got_signal or past_deadline:
            self.stop_reason = "signal" if got_signal else "stop_at deadline"
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                print(
                    f"[GUARD] Stopping at step {state.global_step} "
                    f"({self.stop_reason}). Re-run the same command to resume."
                )
        elif interval_due:
            control.should_save = True

        return control

    def on_save(self, args, state, control, **kwargs):
        self.last_save = time.time()


class CheckpointSync(TrainerCallback):
    """
    Mirror the run dir to `checkpoint_sync_target` with rsync (rank 0 only)
    after every complete checkpoint, so progress survives losing the
    machine's local disk. The target may be a local/mounted path or
    user@host:/path (needs passwordless SSH). Runs in the background;
    a save that lands while a sync is running is picked up afterwards.
    """

    def __init__(self, target: str, output_base: Path, run_dir: Path, rank: int):
        self.enabled = bool(target) and rank == 0
        self.target = str(target).rstrip("/") + "/" if target else ""
        # "/./" makes rsync -R recreate <model>/<run>/ under the target.
        self.source = f"{output_base.parent}/./{output_base.name}/{run_dir.name}/"
        self.proc = None
        self.pending = False

    def _command(self):
        return [
            "rsync", "-aR", "--delete",
            "-e", "ssh -o BatchMode=yes",
            self.source, self.target,
        ]

    def _poll(self):
        if self.proc is None or self.proc.poll() is None:
            return
        if self.proc.returncode != 0:
            print(f"[SYNC] WARNING: rsync exited with code {self.proc.returncode}")
        self.proc = None

    def _start(self):
        print(f"[SYNC] {self.source} -> {self.target}")
        self.proc = subprocess.Popen(self._command())
        self.pending = False

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self._poll()
        if self.proc is None:
            self._start()
        else:
            self.pending = True

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        self._poll()
        if self.pending and self.proc is None:
            self._start()

    def final_sync(self):
        """Blocking sync at exit (last checkpoint, final model, metrics)."""
        if not self.enabled:
            return
        if self.proc is not None:
            self.proc.wait()
        print(f"[SYNC] Final sync -> {self.target}")
        returncode = subprocess.run(self._command()).returncode
        if returncode != 0:
            print(f"[SYNC] WARNING: final rsync exited with code {returncode}")


def cast_adapter_to_bf16(adapter_dir: Path):
    """
    PEFT keeps LoRA weights in fp32; for evaluation bf16 is enough (the base
    model is bf16) and halves the file. Resume checkpoints are not touched.
    """
    from safetensors.torch import load_file, save_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        return
    tensors = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_file(str(path)).items()
    }
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    tmp.replace(path)


def export_final_model(trainer, final_dir: Path, tokenizer_path: Path,
                       label: str = "[FINAL] Model saved"):
    """
    Save a plain HF model (full FT) or adapter (LoRA/QLoRA) to final_dir.

    The ZeRO-3 configs keep stage3_gather_16bit_weights_on_model_save=false
    so periodic checkpoints stay sharded; save_model() would then write yet
    another ZeRO checkpoint. Gather the 16-bit weights once here instead
    (collective: runs on every rank, assembled on rank 0's CPU).
    """
    if trainer.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
        if isinstance(trainer.model, PeftModel):
            # Adapter only: gather just the trainable (LoRA) parameters
            # instead of consolidating the whole frozen base model.
            import deepspeed

            trainable = [p for p in trainer.model.parameters() if p.requires_grad]
            with deepspeed.zero.GatheredParameters(trainable, modifier_rank=None):
                if trainer.args.should_save:
                    trainer.model.save_pretrained(str(final_dir))
            if trainer.args.should_save:
                cast_adapter_to_bf16(final_dir)
        else:
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
            if trainer.args.should_save:
                trainer._save(str(final_dir), state_dict=state_dict)
    else:
        trainer.save_model(str(final_dir))

    if trainer.args.should_save:
        AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True,
        ).save_pretrained(str(final_dir))
        print(f"{label}: {final_dir}")


def export_milestone_weights(trainer, run_dir: Path, percent, tokenizer_path: Path):
    """
    Weights-only milestone (plain HF model / adapter, for evaluation) in
    run_dir/milestone-<pct>pct-step<N>/. Collective: call on every rank.
    """
    step = trainer.state.global_step
    out = run_dir / f"milestone-{int(percent):03d}pct-step{step}"
    if (out / CHECKPOINT_COMPLETE_MARKER).exists():
        return  # already exported at this step (e.g. a stop save on a milestone)
    if trainer.args.should_save and out.exists():
        shutil.rmtree(out)  # left over from an export that was cut off
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    export_final_model(
        trainer, out, tokenizer_path,
        label=f"[CKPT] Saved milestone {percent}% (weights only)",
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if trainer.args.should_save:
        with (out / MILESTONE_FILE).open("w", encoding="utf-8") as f:
            json.dump(
                {"percent": percent, "step": step, "max_steps": trainer.state.max_steps,
                 "epoch": trainer.state.epoch, "weights_only": True},
                f, indent=2,
            )
        (out / CHECKPOINT_COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")


class SteadyStateTimer(TrainerCallback):
    """
    Times steady-state steps of this session.

    Warmup is counted from the step training (re)started at, and time spent
    writing checkpoints is excluded so saves do not skew throughput.
    """

    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self.initial_step = 0
        self.start_step = None
        self.start_time = None
        self.end_step = None
        self.elapsed = None
        self.last_step_end = None
        self.save_seconds = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        # Non-zero when resuming from a checkpoint.
        self.initial_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_step_end = time.perf_counter()

        if (
            self.start_time is None
            and state.global_step - self.initial_step >= self.warmup_steps
        ):
            self.start_step = state.global_step
            self.start_time = self.last_step_end

    def on_save(self, args, state, control, **kwargs):
        if self.start_time is not None and self.last_step_end is not None:
            self.save_seconds += time.perf_counter() - self.last_step_end

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_step = state.global_step
        self.elapsed = (
            time.perf_counter() - self.start_time - self.save_seconds
        )

    @property
    def measured_steps(self):
        if self.start_step is None or self.end_step is None:
            return 0
        return max(0, self.end_step - self.start_step)


def distributed_max(value: float):
    if not (dist.is_available() and dist.is_initialized()):
        return value

    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else "cpu"
    )

    tensor = torch.tensor(
        [value],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument(
        "--deepspeed_config",
        default=None,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Resume from this checkpoint instead of the newest one in output_dir.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start from scratch even if config.yaml has resume: true.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Only check that checkpoint_root allows this run (exit 0/2).",
    )
    parser.add_argument(
        "--stop_at",
        default=None,
        help=(
            "End of the GPU reservation in local time, e.g. '2026-10-09 18:00'. "
            "Training saves and stops stop_margin_minutes before it."
        ),
    )
    return parser.parse_args()


def prepare_run(cfg, args, run_name_prefix: str = ""):
    """
    Setup shared by the BF16 and QLoRA benchmarks:
    token cache, dataset, and TrainingArguments.

    Must be called before from_pretrained() (required for ZeRO-3).
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # --------------------------------------------------------------
    # 1. Rank 0 tokenizes/packs once. Other ranks wait.
    # --------------------------------------------------------------
    cache_metadata = build_cache_if_needed(
        cfg,
        rank=rank,
        rebuild_cache=args.rebuild_cache,
    )

    cache_path = resolve_path(cfg["cache_path"])
    dataset = load_from_disk(str(cache_path))

    seq_len = int(cfg.get("seq_len", 2048))
    dataset_tokens = int(
        cache_metadata.get("total_tokens", len(dataset) * seq_len)
    )

    # --------------------------------------------------------------
    # 2. Training settings
    # --------------------------------------------------------------
    output_base = resolve_output_base(cfg)

    attn_impl = (
        args.attn_implementation
        or cfg.get("attn_implementation", "flash_attention_2")
    )

    # "none" / null disables DeepSpeed (e.g. single GPU or native Windows).
    ds_value = args.deepspeed_config or cfg.get("deepspeed_config")
    ds_path = (
        None
        if ds_value in (None, "", "none")
        else resolve_path(ds_value)
    )

    micro_batch = int(cfg.get("micro_batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else cfg.get("max_steps", 300)
    )
    warmup_steps = int(cfg.get("benchmark_warmup_steps", 20))

    if max_steps > 0 and max_steps <= warmup_steps:
        raise ValueError(
            "max_steps must be larger than benchmark_warmup_steps."
        )

    set_seed(int(cfg.get("seed", 42)))

    run_dir = run_dir_for(cfg, args, run_name_prefix, world_size)
    check_checkpoint_root(cfg, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Checkpointing: optimizer/scheduler/RNG/DeepSpeed state is saved too,
    # so an interrupted run continues exactly where it stopped.
    save_steps = int(cfg.get("save_steps", 0))
    save_total_limit = int(cfg.get("save_total_limit", 1))

    # Stored in every checkpoint and compared on resume (see DATA_ORDER_KEYS).
    data_state = {
        "dataset_signature": cache_metadata.get("signature"),
        "num_sequences": len(dataset),
        "shuffle_documents": bool(cfg.get("shuffle_documents", True)),
        "data_seed": int(cfg.get("data_seed", 0)),
        "seed": int(cfg.get("seed", 42)),
        "world_size": world_size,
        "micro_batch_size": int(cfg.get("micro_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "seq_len": seq_len,
    }

    if args.resume_from_checkpoint:
        resume_checkpoint = resolve_path(args.resume_from_checkpoint)
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_checkpoint}"
            )
    elif bool(cfg.get("resume", True)) and not args.no_resume:
        resume_checkpoint = find_resume_checkpoint(run_dir, rank)
    else:
        resume_checkpoint = None
        if list_checkpoints(run_dir):
            # The first save of a fresh run would delete the previous run's
            # checkpoints; make that an explicit decision.
            raise RuntimeError(
                f"Starting from scratch, but checkpoints already exist in "
                f"{run_dir}. Move or delete them first."
            )

    if resume_checkpoint is not None:
        check_data_state(resume_checkpoint, data_state, rank)

    stop_at = datetime.fromisoformat(args.stop_at) if args.stop_at else None
    if stop_at is not None and stop_at <= datetime.now():
        raise ValueError(f"--stop_at is in the past: {stop_at}")

    # Important for ZeRO-3:
    # construct TrainingArguments (and therefore DeepSpeed integration)
    # before from_pretrained().
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        optim=cfg.get("optim", "adamw_torch"),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_kwargs=(
            {"min_lr_rate": float(cfg["min_lr_rate"])}
            if cfg.get("lr_scheduler_type") == "cosine_with_min_lr"
            and cfg.get("min_lr_rate") is not None
            else {}
        ),
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
        bf16=bool(cfg.get("bf16", True)),
        tf32=bool(cfg.get("tf32", True)),
        gradient_checkpointing=bool(
            cfg.get("gradient_checkpointing", True)
        ),
        # Trainer re-enables checkpointing with these kwargs; without them
        # it falls back to use_reentrant=True.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=save_steps if save_steps > 0 else 500,
        # Old checkpoints are removed by CheckpointCompleteMarker instead.
        save_total_limit=None,
        # Resume skips the batches already trained on, so the data
        # continues where it stopped (never set this to True).
        ignore_data_skip=False,
        # One fixed dir, so curves continue across resumed sessions.
        report_to=cfg.get("report_to", "none") or "none",
        logging_dir=str(run_dir / "tensorboard"),
        remove_unused_columns=False,
        dataloader_num_workers=int(
            cfg.get("dataloader_num_workers", 4)
        ),
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        deepspeed=str(ds_path) if ds_path else None,
        seed=int(cfg.get("seed", 42)),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    return SimpleNamespace(
        rank=rank,
        world_size=world_size,
        cache_metadata=cache_metadata,
        cache_path=cache_path,
        dataset=dataset,
        dataset_tokens=dataset_tokens,
        seq_len=seq_len,
        attn_impl=attn_impl,
        ds_path=ds_path,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        output_base=output_base,
        run_dir=run_dir,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        resume_checkpoint=resume_checkpoint,
        data_state=data_state,
        stop_at=stop_at,
        training_args=training_args,
    )


def print_run_header(title: str, cfg, run, model_path: Path, extra=None):
    if run.rank != 0:
        return

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Model                 : {model_path}")
    for label, value in (extra or {}).items():
        print(f"{label:<22}: {value}")
    print(f"Raw data              : {resolve_path(cfg['data_path'])}")
    print(f"Token cache           : {run.cache_path}")
    print(f"Dataset tokens        : {run.dataset_tokens:,}")
    print(f"Attention             : {run.attn_impl}")
    print(f"DeepSpeed             : {run.ds_path}")
    print(f"World size            : {run.world_size}")
    print(f"Sequence length       : {run.seq_len}")
    print(f"Micro batch / GPU     : {run.micro_batch}")
    print(f"Gradient accumulation : {run.grad_accum}")
    print(f"Max optimizer steps   : {run.max_steps}")
    print(
        f"LR schedule           : {cfg.get('learning_rate')} "
        f"{cfg.get('lr_scheduler_type', 'linear')}, warmup_ratio {cfg.get('warmup_ratio', 0.0)}, "
        f"min_lr_rate {cfg.get('min_lr_rate')}"
    )
    print(f"Output dir            : {run.run_dir}")
    if run.save_steps > 0:
        keep = run.save_total_limit if run.save_total_limit > 0 else "all"
        print(f"Checkpoint every      : {run.save_steps} steps (keep {keep})")
    else:
        print("Checkpoint every      : disabled")
    interval = float(cfg.get("save_interval_minutes", 0))
    if interval > 0:
        print(f"Checkpoint every      : {interval:g} min")
    print(f"Resume from           : {run.resume_checkpoint or 'scratch'}")
    if run.stop_at is not None:
        margin = float(cfg.get("stop_margin_minutes", 10))
        print(f"Stop at               : {run.stop_at} (save {margin:g} min before)")
    print(f"Checkpoint sync       : {cfg.get('checkpoint_sync_target') or 'disabled'}")
    print("=" * 80)


def build_lora_config(cfg):
    return LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias=cfg.get("lora_bias", "none"),
        target_modules=cfg.get(
            "lora_target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    args = parse_args(
        "Integrated tokenize/cache + BF16 (full / LoRA) DeepSpeed benchmark."
    )
    cfg = load_model_config(resolve_path(args.config), args.model)
    preflight(cfg, args)
    run = prepare_run(cfg, args)

    model_path = resolve_path(cfg["model_path"])
    attn_impl = run.attn_impl
    rank = run.rank

    print_run_header(
        "FULL TRAINING BENCHMARK", cfg, run, model_path,
        {"Training mode": cfg.get("training_mode", "fft")},
    )

    # --------------------------------------------------------------
    # 3. Model load
    # --------------------------------------------------------------
    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    if bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

    model_load_seconds = time.perf_counter() - load_start

    if cfg.get("training_mode", "fft") == "lora":
        model = get_peft_model(model, build_lora_config(cfg))

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if rank == 0:
            model.print_trainable_parameters()

    train_and_report(
        cfg, run, model, model_load_seconds,
        {"precision": "bf16", "training_mode": cfg.get("training_mode", "fft")},
    )


class CPTTrainer(Trainer):
    """
    Trainer that drops empty optimizer param groups. With LoRA every
    trainable parameter takes weight decay, so the no-decay group is empty;
    DeepSpeed ZeRO-3 removes it, leaving the LR scheduler with one more
    group than the optimizer (fails on the first scheduler step).
    """

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        # With an "optimizer" section in the DeepSpeed config (e.g. zero2.json)
        # this is a placeholder without param_groups; DeepSpeed builds it.
        if hasattr(optimizer, "param_groups"):
            optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]
        return optimizer


def train_and_report(cfg, run, model, model_load_seconds: float, extra_metrics=None):
    rank = run.rank
    world_size = run.world_size
    cache_metadata = run.cache_metadata
    dataset = run.dataset
    dataset_tokens = run.dataset_tokens
    seq_len = run.seq_len
    attn_impl = run.attn_impl
    micro_batch = run.micro_batch
    grad_accum = run.grad_accum
    warmup_steps = run.warmup_steps
    run_dir = run.run_dir
    training_args = run.training_args

    # --------------------------------------------------------------
    # 4. Actual training benchmark:
    #    forward -> NTP loss -> backward -> optimizer -> ZeRO comm
    # --------------------------------------------------------------
    timer = SteadyStateTimer(warmup_steps)
    guard = InterruptionGuard(
        save_interval_minutes=float(cfg.get("save_interval_minutes", 0)),
        stop_at=run.stop_at,
        stop_margin_minutes=float(cfg.get("stop_margin_minutes", 10)),
    )
    sync = CheckpointSync(
        cfg.get("checkpoint_sync_target"), run.output_base, run_dir, rank
    )
    weights_only = bool(cfg.get("milestone_weights_only", False))
    tokenizer_path = resolve_path(cfg["model_path"])
    milestones = MilestoneSaver(
        cfg.get("milestone_percents"), weights_only=weights_only, tokenizer_path=tokenizer_path
    )
    marker = CheckpointCompleteMarker(
        run.save_total_limit, run.data_state, milestones,
        weights_only=weights_only, tokenizer_path=tokenizer_path,
    )

    trainer = CPTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(),
        # Order matters: the marker (and old-checkpoint removal) must finish
        # before the timer measures the save and before the sync starts.
        callbacks=[
            milestones,
            marker,
            timer,
            guard,
            sync,
        ],
    )

    marker.trainer = trainer
    milestones.trainer = trainer

    resume_checkpoint = run.resume_checkpoint
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint else None
        )
    )
    # Runtime and steps of this session only (not of earlier, resumed ones).
    train_runtime = float(
        train_result.metrics.get("train_runtime", 0.0)
    )
    session_steps = trainer.state.global_step - timer.initial_step

    tokens_per_optimizer_step = (
        world_size
        * micro_batch
        * grad_accum
        * seq_len
    )

    processed_tokens = (
        session_steps
        * tokens_per_optimizer_step
    )

    overall_tps = (
        processed_tokens / train_runtime
        if train_runtime > 0
        else 0.0
    )

    if timer.measured_steps > 0 and timer.elapsed:
        steady_tokens = (
            timer.measured_steps
            * tokens_per_optimizer_step
        )
        steady_tps = steady_tokens / timer.elapsed
        steady_step_seconds = (
            timer.elapsed / timer.measured_steps
        )
    else:
        steady_tps = overall_tps
        steady_step_seconds = (
            train_runtime / max(1, session_steps)
        )

    projected_epoch_hours = (
        dataset_tokens / steady_tps / 3600
        if steady_tps > 0
        else None
    )

    peak_allocated_gb = distributed_max(
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    peak_reserved_gb = distributed_max(
        torch.cuda.max_memory_reserved() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    process_rss_gb = distributed_max(
        psutil.Process(os.getpid()).memory_info().rss
        / (1024 ** 3)
    )

    # After the memory measurements, so the export does not skew them.
    # Only a run that reached its end gets a final model; an interrupted one
    # is continued from its checkpoint by re-running the same command.
    final_dir = None
    if guard.stop_reason is None and bool(cfg.get("export_final_model", True)):
        # The 100% weights-only milestone already is the final model.
        done = [p for p in run_dir.glob("milestone-100pct-step*") if milestone_percent(p) == 100]
        if done:
            final_dir = done[-1]
        else:
            final_dir = run_dir / "final"
            export_final_model(trainer, final_dir, resolve_path(cfg["model_path"]))

    metrics = {
        "name": cfg["name"],
        **(extra_metrics or {}),
        "attention": attn_impl,
        "world_size": world_size,
        "sequence_length": seq_len,
        "micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "completed_optimizer_steps": trainer.state.global_step,
        "resumed_from_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint else None
        ),
        "session_optimizer_steps": session_steps,
        "checkpoint_save_seconds": timer.save_seconds,
        "stopped_early": guard.stop_reason,
        "final_model_dir": str(final_dir) if final_dir else None,
        "preprocessing_seconds": cache_metadata.get(
            "preprocessing_seconds"
        ),
        "model_load_seconds": model_load_seconds,
        "training_runtime_seconds": train_runtime,
        "overall_tokens_per_second": overall_tps,
        "steady_state_tokens_per_second": steady_tps,
        "steady_state_seconds_per_step": steady_step_seconds,
        "dataset_tokens": dataset_tokens,
        "projected_one_epoch_hours": projected_epoch_hours,
        "max_peak_allocated_gpu_gb": peak_allocated_gb,
        "max_peak_reserved_gpu_gb": peak_reserved_gb,
        "max_rank_process_rss_gb": process_rss_gb,
    }

    if rank == 0:
        metrics_path = run_dir / "benchmark_metrics.json"

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 80)
        print("BENCHMARK RESULT")
        print("=" * 80)
        print(
            "Preprocessing       : "
            f"{cache_metadata.get('preprocessing_seconds', 0):.2f} sec "
            "(0 additional sec when cache is reused)"
        )
        print(f"Model load          : {model_load_seconds:.2f} sec")
        print(f"Training runtime    : {train_runtime:.2f} sec")
        print(f"Steady throughput   : {steady_tps:,.1f} tokens/sec")
        print(f"Steady step time    : {steady_step_seconds:.3f} sec")
        print(f"Peak GPU allocated  : {peak_allocated_gb:.2f} GiB")
        print(f"Peak GPU reserved   : {peak_reserved_gb:.2f} GiB")
        print(f"Max rank CPU RSS    : {process_rss_gb:.2f} GiB")
        if projected_epoch_hours is not None:
            print(
                f"Projected 1 epoch   : "
                f"{projected_epoch_hours:.2f} hours"
            )
        print(f"Metrics JSON        : {metrics_path}")
        if guard.stop_reason is not None:
            print(f"Stopped early       : {guard.stop_reason} "
                  f"(step {trainer.state.global_step}; re-run to resume)")
        elif final_dir is not None:
            print(f"Final model         : {final_dir}")
        print("=" * 80)

    sync.final_sync()


if __name__ == "__main__":
    main()
