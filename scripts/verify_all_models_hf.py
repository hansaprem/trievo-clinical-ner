"""
Verify all 4 production checkpoints on Hugging Face Hub.
Checks remote files, confirms model.safetensors presence and size (~415 MB) for each.
"""

from huggingface_hub import HfApi

ALL_REPOS = [
    "hansaprem703/trievo-bc5cdr-pubmedbert",
    "hansaprem703/trievo-ncbi-disease-pubmedbert",
    "hansaprem703/trievo-jnlpba-pubmedbert",
    "hansaprem703/trievo-anatem-pubmedbert",
]

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


def verify_all():
    api = HfApi()
    overall_ok = True

    for repo_id in ALL_REPOS:
        print(f"\n{'='*55}")
        print(f"Inspecting repository: {repo_id}")
        print(f"{'='*55}")

        try:
            info = api.repo_info(repo_id, repo_type="model")
            print(f"Accessibility: Public (private={info.private})")
        except Exception as e:
            print(f"Error accessing repo: {e}")
            overall_ok = False
            continue

        files = api.list_repo_tree(repo_id, repo_type="model")
        remote_files = {f.path: f.size for f in files}

        print("Remote files found:")
        for path, size in remote_files.items():
            size_str = f"{size:,} bytes" if size is not None else "N/A"
            print(f"  - {path}: {size_str}")

        print("Checking required files:")
        repo_ok = True
        for req in REQUIRED_FILES:
            if req in remote_files:
                print(f"  [OK] {req}")
            else:
                print(f"  [MISSING] {req}")
                repo_ok = False
                overall_ok = False

        if "model.safetensors" in remote_files:
            msize = remote_files["model.safetensors"]
            msize_mb = msize / (1024 * 1024) if msize else 0
            print(f"model.safetensors size: {msize_mb:.2f} MB ({msize:,} bytes)")
            if 410 <= msize_mb <= 420:
                print("  [OK] Remote size matches expected ~415 MB.")
            else:
                print(f"  [WARNING] Size {msize_mb:.2f} MB is outside expected ~415 MB.")
                overall_ok = False
        else:
            print("  [ERROR] model.safetensors not found.")
            overall_ok = False

    print(f"\n==========================================")
    if overall_ok:
        print("ALL 4 MODEL REPOSITORIES 100% VERIFIED!")
    else:
        print("SOME CHECKS FAILED OR INCOMPLETE.")
    print(f"==========================================")
    return overall_ok


if __name__ == "__main__":
    verify_all()
