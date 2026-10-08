"""
Pre-quantize BF16 checkpoints to 4-bit (bitsandbytes) and save them to
`quantized_model_path` from config.yaml.

Needs CUDA GPU(s): bitsandbytes quantizes on the GPU while loading.

Usage:
  python scripts/quantize_models.py --model qwen3_8b
  python scripts/quantize_models.py --model all
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from train_benchmark import load_model_config, resolve_path  # noqa: E402
from train_qlora import build_bnb_config  # noqa: E402


with CONFIG_PATH.open("r", encoding="utf-8") as f:
    MODEL_KEYS = [
        key
        for key, info in yaml.safe_load(f)["models"].items()
        if info.get("quantized_model_path")
    ]


def quantize_model(model_key):
    cfg = load_model_config(CONFIG_PATH, model_key, qlora=True)

    src = resolve_path(cfg["model_path"])
    dst = resolve_path(cfg["quantized_model_path"])

    print()
    print("=" * 70)
    print(f"Model        : {model_key}")
    print(f"BF16 source  : {src}")
    print(f"4-bit output : {dst}")
    print(f"Quantization : {cfg['quantization']}")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=build_bnb_config(cfg["quantization"]),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )

    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        src,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(dst)

    print()
    print(f"[DONE] {model_key}")
    print(f"Saved to: {dst}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-quantize BF16 checkpoints to 4-bit (bitsandbytes)."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*MODEL_KEYS, "all"],
        help="Model to quantize.",
    )
    args = parser.parse_args()

    if args.model == "all":
        # Several entries (e.g. full + LoRA) can share one checkpoint.
        seen = set()
        for model_key in MODEL_KEYS:
            dst = load_model_config(CONFIG_PATH, model_key, qlora=True)["quantized_model_path"]
            if dst in seen:
                continue
            seen.add(dst)
            quantize_model(model_key)
    else:
        quantize_model(args.model)


if __name__ == "__main__":
    main()
