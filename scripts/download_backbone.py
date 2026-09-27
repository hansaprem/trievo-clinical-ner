import os
import sys
import time
import urllib.request
from pathlib import Path

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"Already exists: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return
        
    print(f"Downloading {url} -> {dest_path.name} ...")
    temp_path = dest_path.with_suffix(".part")
    t0 = time.time()
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_log = t0
        while True:
            chunk = resp.read(1024 * 1024 * 4) # 4MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log > 3.0:
                pct = (downloaded / total * 100) if total else 0
                speed = (downloaded / (1024 * 1024)) / (now - t0 + 1e-5)
                print(f"  {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({pct:.1f}%) - {speed:.2f} MB/s", flush=True)
                last_log = now
                
    temp_path.replace(dest_path)
    dur = time.time() - t0
    print(f"Completed {dest_path.name}: {downloaded / (1024*1024):.1f} MB in {dur:.1f}s ({downloaded/(1024*1024)/dur:.2f} MB/s)")


def download_repo(repo_id: str, dest_dir: Path, files: list):
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    for fname in files:
        url = f"{base_url}/{fname}"
        dest = dest_dir / fname
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Error downloading {fname}: {e}")

if __name__ == "__main__":
    out_dir = Path(r"C:\Users\khan computer\.gemini\antigravity\scratch\trievo-clinical-ner\pretrained_backbones\bert-base-uncased")
    download_repo("google-bert/bert-base-uncased", out_dir, ["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
