"""
Delete everything under checkpoint_root (config.yaml) to make room for the
next fine-tuning. Run it by hand, after the milestone checkpoints have been
copied off this machine. Nothing in the pipeline calls this script.

    python scripts/clear_checkpoints.py            # list, then ask
    python scripts/clear_checkpoints.py --model qwen3_8b

It only deletes after you type DELETE at the prompt, and refuses to run
without a terminal or while training is running.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import psutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def resolve_path(path_value):
    p = Path(path_value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def dir_size(path: Path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def training_processes():
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info["cmdline"] or [])
        if "train_benchmark.py" in cmd or "train_qlora.py" in cmd:
            found.append(f"  PID {proc.info['pid']}: {cmd[:120]}")
    return found


def describe(run_dir: Path):
    lines = []
    for checkpoint in sorted(run_dir.glob("checkpoint-*")) + sorted(run_dir.glob("milestone-*")):
        milestone = checkpoint / "MILESTONE.json"
        if milestone.exists():
            with milestone.open("r", encoding="utf-8") as f:
                info = json.load(f)
            kind = "weights" if info.get("weights_only") else "full state"
            tag = f"milestone {info['percent']}% ({kind})"
        else:
            tag = "resume checkpoint"
        lines.append(f"      {checkpoint.name:<28} {tag}")
    if (run_dir / "final").is_dir():
        lines.append("      final                  exported model")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--model", help="Only clear this model's output_dir.")
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root_value = cfg["defaults"].get("checkpoint_root")
    if not root_value:
        sys.exit("checkpoint_root is not set in config.yaml.")
    root = resolve_path(root_value)

    if args.model:
        out = Path(cfg["models"][args.model]["output_dir"])
        targets = [out if out.is_absolute() else root / out]
    else:
        targets = sorted(p for p in root.iterdir()) if root.is_dir() else []
    targets = [t for t in targets if t.exists()]

    if not targets:
        print(f"Nothing to clear under {root}.")
        return

    busy = training_processes()
    if busy:
        sys.exit("Training is running; stop it first:\n" + "\n".join(busy))

    if not sys.stdin.isatty():
        sys.exit("Refusing to delete without an interactive terminal.")

    total = 0
    print(f"Will delete under {root}:")
    for target in targets:
        size = dir_size(target) if target.is_dir() else target.stat().st_size
        total += size
        print(f"  {target.name}  ({size / 1024**3:,.1f} GiB)")
        for run_dir in sorted(p for p in target.iterdir() if p.is_dir()) if target.is_dir() else []:
            print(f"    {run_dir.name}")
            for line in describe(run_dir):
                print(line)
    print(f"Total: {total / 1024**3:,.1f} GiB")
    print("Make sure the milestone checkpoints have been copied off this machine.")

    if input("Type DELETE to delete them: ").strip() != "DELETE":
        print("Cancelled; nothing deleted.")
        return

    for target in targets:
        print(f"Deleting {target}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
