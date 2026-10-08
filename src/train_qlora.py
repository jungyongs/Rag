"""
QLoRA benchmark: 4-bit (bitsandbytes) base model + trainable LoRA adapters.

Counterpart of train_benchmark.py (BF16 full / LoRA). 4-bit weights are
frozen, so only the LoRA adapters are trained. The token cache, collator,
timer and metrics code are shared with train_benchmark.py.

Where the 4-bit base model comes from (config.yaml -> qlora.use_prequantized):
  false : quantize the BF16 checkpoint at `model_path` while loading.
  true  : load the checkpoint at `quantized_model_path`, written beforehand by
          scripts/quantize_models.py.
"""
import os
import time

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from train_benchmark import (
    build_lora_config,
    load_model_config,
    parse_args,
    prepare_run,
    print_run_header,
    resolve_path,
    train_and_report,
)


def build_bnb_config(qcfg):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(
            qcfg.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=getattr(
            torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
        # Must match the dtype of the other (non-quantized) params so that
        # ZeRO-3 can flatten/shard everything together.
        bnb_4bit_quant_storage=getattr(
            torch, qcfg.get("bnb_4bit_quant_storage", "bfloat16")
        ),
    )


def resolve_base_model(cfg):
    """Return (checkpoint path, BitsAndBytesConfig or None if pre-quantized)."""
    if bool(cfg.get("use_prequantized", False)):
        path = resolve_path(cfg["quantized_model_path"])
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"Pre-quantized checkpoint not found: {path}\n"
                f"Run: python scripts/quantize_models.py --model {cfg['name']}"
            )
        # The quantization config is stored inside the checkpoint.
        return path, None

    return resolve_path(cfg["model_path"]), build_bnb_config(cfg["quantization"])


def main():
    args = parse_args(
        "Integrated tokenize/cache + 4-bit QLoRA DeepSpeed benchmark."
    )
    cfg = load_model_config(resolve_path(args.config), args.model, qlora=True)

    # Builds TrainingArguments, so DeepSpeed (incl. ZeRO-3) is configured
    # before from_pretrained().
    run = prepare_run(cfg, args, run_name_prefix="qlora_")

    model_path, bnb_config = resolve_base_model(cfg)
    zero3 = is_deepspeed_zero3_enabled()
    qcfg = cfg.get("quantization", {})
    quant_desc = (
        "pre-quantized checkpoint"
        if bnb_config is None
        else f"on-the-fly {qcfg.get('bnb_4bit_quant_type', 'nf4')}"
    )

    print_run_header(
        "QLoRA (4-BIT) TRAINING BENCHMARK", cfg, run, model_path,
        {"Quantization": quant_desc, "ZeRO-3": zero3},
    )

    # --------------------------------------------------------------
    # 3. 4-bit model load
    # --------------------------------------------------------------
    load_start = time.perf_counter()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=run.attn_impl,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    if not zero3:
        # DDP / ZeRO-1/2: every rank holds the full 4-bit model on its own GPU.
        # ZeRO-3 places and shards the weights itself.
        load_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False

    model_load_seconds = time.perf_counter() - load_start

    use_gc = bool(cfg.get("gradient_checkpointing", True))
    gc_kwargs = {"use_reentrant": False}

    if zero3:
        # prepare_model_for_kbit_training() upcasts non-quantized params to
        # fp32, which breaks ZeRO-3's uniform-dtype flattening. Do only the
        # parts that are still needed.
        if use_gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()
    else:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gc,
            gradient_checkpointing_kwargs=gc_kwargs,
        )

    model = get_peft_model(model, build_lora_config(cfg))

    if run.rank == 0:
        model.print_trainable_parameters()

    train_and_report(
        cfg, run, model, model_load_seconds,
        {
            "precision": "4bit",
            "training_mode": "qlora",
            "quantization": quant_desc,
            "bnb_4bit_quant_type": qcfg.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_use_double_quant": qcfg.get("bnb_4bit_use_double_quant", True),
            "base_checkpoint": str(model_path),
        },
    )


if __name__ == "__main__":
    main()
