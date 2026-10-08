import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

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
from peft import LoraConfig, TaskType, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


class SteadyStateTimer(TrainerCallback):
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self.start_step = None
        self.start_time = None
        self.end_step = None
        self.elapsed = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None and state.global_step >= self.warmup_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_step = state.global_step
            self.start_time = time.perf_counter()

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_step = state.global_step
        self.elapsed = time.perf_counter() - self.start_time

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


def main():
    parser = argparse.ArgumentParser(
        description="Integrated tokenize/cache + full-parameter DeepSpeed benchmark."
    )
    parser.add_argument("--config", required=True)
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
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    optim=cfg.get("optim", "adamw_torch"),

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
    model_path = resolve_path(cfg["model_path"])
    output_base = resolve_path(cfg["output_dir"])

    attn_impl = (
        args.attn_implementation
        or cfg.get("attn_implementation", "flash_attention_2")
    )

    ds_path = resolve_path(
        args.deepspeed_config or cfg["deepspeed_config"]
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

    run_dir = output_base / f"{attn_impl}_{world_size}gpu"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Important for ZeRO-3:
    # construct TrainingArguments (and therefore DeepSpeed integration)
    # before from_pretrained().
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
        bf16=bool(cfg.get("bf16", True)),
        tf32=bool(cfg.get("tf32", True)),
        gradient_checkpointing=bool(
            cfg.get("gradient_checkpointing", True)
        ),
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(
            cfg.get("dataloader_num_workers", 4)
        ),
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        deepspeed=str(ds_path),
        seed=int(cfg.get("seed", 42)),
    )

    if rank == 0:
        print("\n" + "=" * 80)
        print("FULL TRAINING BENCHMARK")
        print("=" * 80)
        print(f"Model                 : {model_path}")
        print(f"Raw data              : {resolve_path(cfg['data_path'])}")
        print(f"Token cache           : {cache_path}")
        print(f"Dataset tokens        : {dataset_tokens:,}")
        print(f"Attention             : {attn_impl}")
        print(f"DeepSpeed             : {ds_path}")
        print(f"World size            : {world_size}")
        print(f"Sequence length       : {seq_len}")
        print(f"Micro batch / GPU     : {micro_batch}")
        print(f"Gradient accumulation : {grad_accum}")
        print(f"Max optimizer steps   : {max_steps}")
        print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

        lora_config = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
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

        model = get_peft_model(model, lora_config)

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if rank == 0:
            model.print_trainable_parameters()
    # --------------------------------------------------------------
    # 4. Actual training benchmark:
    #    forward -> NTP loss -> backward -> optimizer -> ZeRO comm
    # --------------------------------------------------------------
    timer = SteadyStateTimer(warmup_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(),
        callbacks=[timer],
    )

    train_result = trainer.train()
    train_runtime = float(
        train_result.metrics.get("train_runtime", 0.0)
    )

    tokens_per_optimizer_step = (
        world_size
        * micro_batch
        * grad_accum
        * seq_len
    )

    processed_tokens = (
        trainer.state.global_step
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
            train_runtime / max(1, trainer.state.global_step)
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

    metrics = {
        "name": cfg["name"],
        "attention": attn_impl,
        "world_size": world_size,
        "sequence_length": seq_len,
        "micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "completed_optimizer_steps": trainer.state.global_step,
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
        print("=" * 80)


if __name__ == "__main__":
    main()
