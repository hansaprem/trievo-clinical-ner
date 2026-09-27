"""
Upload BC5CDR production checkpoint to Hugging Face Hub.
Uses system-cached authentication from `python -m huggingface_hub.cli.hf auth login`.
NEVER logs, prints, or stores credentials or tokens.
"""

from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "hansaprem703/trievo-bc5cdr-pubmedbert"
LOCAL_DIR = Path("models/bc5cdr_pubmedbert")

# Strictly specified production files
PRODUCTION_FILES = [
    "model.safetensors",
    "config.json",
    "label.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "metrics.json",
    "model_card.md",
]


def main():
    api = HfApi()
    user_info = api.whoami()
    print(f"Authenticated as Hugging Face user: {user_info.get('name')}")
    print(f"Target repository: {REPO_ID}")

    for filename in PRODUCTION_FILES:
        filepath = LOCAL_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file not found: {filepath}")
        print(f"Uploading {filename} ({filepath.stat().st_size:,} bytes)...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="model",
        )

    print("All production files uploaded successfully!")


if __name__ == "__main__":
    main()
