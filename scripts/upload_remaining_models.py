"""
Upload the remaining 3 production checkpoints (NCBI Disease, JNLPBA, AnatEM)
to Hugging Face Hub using cached authentication.
NEVER logs, prints, or stores credentials or tokens.
"""

from pathlib import Path
from huggingface_hub import HfApi

MODELS_TO_UPLOAD = {
    "hansaprem703/trievo-ncbi-disease-pubmedbert": Path("models/ncbi_disease_pubmedbert"),
    "hansaprem703/trievo-jnlpba-pubmedbert": Path("models/jnlpba_pubmedbert"),
    "hansaprem703/trievo-anatem-pubmedbert": Path("models/anatem_pubmedbert"),
}

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

    for repo_id, local_dir in MODELS_TO_UPLOAD.items():
        print(f"\n==========================================")
        print(f"Processing repository: {repo_id}")
        print(f"Local source: {local_dir}")
        print(f"==========================================")

        for filename in PRODUCTION_FILES:
            filepath = local_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing required file: {filepath}")
            print(f"Uploading {filename} ({filepath.stat().st_size:,} bytes) -> {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
        print(f"Successfully uploaded all 8 production files for {repo_id}!")

    print("\nAll 3 remaining model repositories uploaded successfully!")


if __name__ == "__main__":
    main()
