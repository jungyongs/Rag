import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


# ---------------------------------------------------------
# Path
# ---------------------------------------------------------
# download_models.py:
#   <PROJECT_ROOT>/scripts/download_models.py
#
# therefore parents[1] == <PROJECT_ROOT>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"


# ---------------------------------------------------------
# Model repositories
# ---------------------------------------------------------
MODELS = {
    "ax4_72b": {
        "repo_id": "skt/A.X-4.0",
        "local_name": "AX-4.0",
    },

    "ax4_light_7b": {
        "repo_id": "skt/A.X-4.0-Light",
        "local_name": "AX-4.0-Light",
    },

    "qwen25_72b": {
        "repo_id": "Qwen/Qwen2.5-72B",
        "local_name": "Qwen2.5-72B",
    },

    "qwen3_8b": {
        "repo_id": "Qwen/Qwen3-8B-Base",
        "local_name": "Qwen3-8B-Base",
    },

    "apertus_70b": {
        "repo_id": "EPFLiGHT/Apertus-70B-MeditronFO",
        "local_name": "Apertus-70B-MeditronFO",
    },
}


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------
def download_model(model_key):
    info = MODELS[model_key]

    save_path = MODEL_ROOT / info["local_name"]

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

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        for model_key in MODELS:
            download_model(model_key)
    else:
        download_model(args.model)


if __name__ == "__main__":
    main()