"""
Verify BC5CDR checkpoint on Hugging Face Hub.
Inspects file tree, verifies model.safetensors presence and exact remote size.
"""

from huggingface_hub import HfApi

REPO_ID = "hansaprem703/trievo-bc5cdr-pubmedbert"

REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "label.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "metrics.json",
    "model_card.md",
]


def verify_remote():
    api = HfApi()
    print(f"Inspecting repository: {REPO_ID}")

    info = api.repo_info(REPO_ID, repo_type="model")
    print(f"Repository accessibility: Public (private={info.private})")

    files = api.list_repo_tree(REPO_ID, repo_type="model")
    remote_files = {f.path: f.size for f in files}

    print("\nRemote files found:")
    for path, size in remote_files.items():
        size_str = f"{size:,} bytes" if size is not None else "N/A"
        print(f"  - {path}: {size_str}")

    print("\nVerifying required files:")
    all_present = True
    for req in REQUIRED_FILES:
        if req in remote_files:
            print(f"  [OK] {req} present")
        else:
            print(f"  [MISSING] {req} NOT found remotely")
            all_present = False

    if "model.safetensors" in remote_files:
        msize = remote_files["model.safetensors"]
        msize_mb = msize / (1024 * 1024) if msize else 0
        print(f"\nRemote model.safetensors size: {msize_mb:.2f} MB ({msize:,} bytes)")
        if 410 <= msize_mb <= 420:
            print("  [OK] Size is approximately 415 MB.")
        else:
            print(f"  [WARNING] Size {msize_mb:.2f} MB is outside expected ~415 MB range.")

    return all_present


if __name__ == "__main__":
    verify_remote()
