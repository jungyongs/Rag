import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


# ---------------------------------------------------------
# Path
# ---------------------------------------------------------
# download_models.py:
#   <PROJECT_ROOT>/scripts/download_models.py
#
# therefore parents[1] == <PROJECT_ROOT>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------
# Model repositories (from config.yaml)
# ---------------------------------------------------------
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    MODELS = {
        key: info
        for key, info in yaml.safe_load(f)["models"].items()
        if info.get("repo_id")
    }


def resolve_path(path_value):
    p = Path(path_value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------
def download_model(model_key):
    info = MODELS[model_key]

    save_path = resolve_path(info["model_path"])

    print()
    print("=" * 70)
    print(f"Model      : {model_key}")
    print(f"Repository : {info['repo_id']}")
    print(f"Save path  : {save_path}")
    print("=" * 70)

    save_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(save_path),
    )

    print()
    print(f"[DONE] {model_key}")
    print(f"Saved to: {save_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from Hugging Face."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[*MODELS.keys(), "all"],
        help="Model checkpoint to download.",
    )

    args = parser.parse_args()

    if args.model == "all":
        # Several entries (e.g. full + LoRA) can share one checkpoint.
        seen = set()
        for model_key, info in MODELS.items():
            if info["model_path"] in seen:
                continue
            seen.add(info["model_path"])
            download_model(model_key)
    else:
        download_model(args.model)


if __name__ == "__main__":
    main()