import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import time
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
from peft import LoraConfig, TaskType, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# Written into checkpoint-<step>/ only after every rank finished saving it.
CHECKPOINT_COMPLETE_MARKER = "_COMPLETE"


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


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


def discover_documents(input_path: Path, pattern: str):
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(p for p in input_path.glob(pattern) if p.is_file())

    raise FileNotFoundError(f"Data path not found: {input_path}")


def dataset_signature(files, model_path: Path, seq_len: int, add_eos: bool):
    """
    Lightweight signature used only to detect obvious cache mismatches.
    It hashes paths, file sizes, mtimes, tokenizer/model path, and key settings.
    It does not hash the entire multi-GB corpus.
    """
    h = hashlib.sha256()
    h.update(str(model_path).encode())
    h.update(str(seq_len).encode())
    h.update(str(add_eos).encode())

    for p in sorted(files):
        stat = p.stat()
        h.update(str(p.resolve()).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())

    return h.hexdigest()


def iter_text_chunks(file_path: Path, chunk_chars: int):
    """
    Stream a large text file instead of reading a 1 GiB file into RAM.

    Chunk boundaries are not treated as document boundaries and do not get EOS.
    EOS is inserted only after the physical file/document ends.
    """
    buffer = []
    char_count = 0

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line) > chunk_chars:
                if buffer:
                    yield "".join(buffer)
                    buffer = []
                    char_count = 0

                for start in range(0, len(line), chunk_chars):
                    yield line[start:start + chunk_chars]
                continue

            buffer.append(line)
            char_count += len(line)

            if char_count >= chunk_chars:
                yield "".join(buffer)
                buffer = []
                char_count = 0

    if buffer:
        yield "".join(buffer)


def packed_generator(
    files,
    tokenizer,
    seq_len,
    add_eos_between_documents,
    chunk_chars,
):
    token_buffer = []
    eos_id = tokenizer.eos_token_id

    for file_path in files:
        for text_chunk in iter_text_chunks(file_path, chunk_chars):
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
    data_seed = int(cfg.get("data_seed", 42))
    add_eos = bool(cfg.get("add_eos_between_documents", True))
    chunk_chars = int(cfg.get("tokenize_chunk_chars", 1_000_000))

    files = discover_documents(data_path, pattern)
    if not files:
        raise RuntimeError(f"No .txt files found at: {data_path}")

    signature = dataset_signature(files, model_path, seq_len, add_eos)

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

        if shuffle_documents:
            rng = random.Random(data_seed)
            rng.shuffle(files)

        print("\n" + "=" * 80)
        print("TOKENIZATION / PACKING (RANK 0 ONLY)")
        print("=" * 80)
        print(f"Tokenizer model : {model_path}")
        print(f"Raw data        : {data_path}")
        print(f"Documents/files : {len(files):,}")
        print(f"Seq length      : {seq_len}")
        print(f"EOS per file    : {add_eos}")
        print(f"Cache           : {cache_path}")
        print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True,
        )

        start = time.perf_counter()

        dataset = Dataset.from_generator(
            packed_generator,
            gen_kwargs={
                "files": files,
                "tokenizer": tokenizer,
                "seq_len": seq_len,
                "add_eos_between_documents": add_eos,
                "chunk_chars": chunk_chars,
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
            "num_documents": len(files),
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

    def __init__(self, keep: int, data_state: dict):
        self.keep = keep
        self.data_state = data_state

    def on_save(self, args, state, control, **kwargs):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

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

        (checkpoint / CHECKPOINT_COMPLETE_MARKER).write_text(
            "complete\n", encoding="utf-8"
        )
        print(f"[CKPT] Saved: {checkpoint}")

        if self.keep > 0:
            others = [c for c in list_checkpoints(run_dir) if c != checkpoint]
            for old in others[: max(0, len(others) - (self.keep - 1))]:
                print(f"[CKPT] Removing old checkpoint: {old}")
                shutil.rmtree(old, ignore_errors=True)


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


def export_final_model(trainer, final_dir: Path, tokenizer_path: Path):
    """
    Save a plain HF model (full FT) or adapter (LoRA/QLoRA) to final_dir.

    The ZeRO-3 configs keep stage3_gather_16bit_weights_on_model_save=false
    so periodic checkpoints stay sharded; save_model() would then write yet
    another ZeRO checkpoint. Gather the 16-bit weights once here instead
    (collective: runs on every rank, assembled on rank 0's CPU).
    """
    if trainer.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
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
        print(f"[FINAL] Model saved: {final_dir}")


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
    output_base = resolve_path(cfg["output_dir"])

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

    run_dir = output_base / f"{run_name_prefix}{attn_impl}_{world_size}gpu"
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
        "data_seed": int(cfg.get("data_seed", 42)),
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(),
        # Order matters: the marker (and old-checkpoint removal) must finish
        # before the timer measures the save and before the sync starts.
        callbacks=[
            CheckpointCompleteMarker(run.save_total_limit, run.data_state),
            timer,
            guard,
            sync,
        ],
    )

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
