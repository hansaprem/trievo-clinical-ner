import os
import sys
import time
import urllib.request
from pathlib import Path

MODELS = [
    {
        "id": "google-bert/bert-base-uncased",
        "slug": "bert-base-uncased",
        "files": ["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    },
    {
        "id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "slug": "pubmedbert",
        "files": ["config.json", "vocab.txt", "tokenizer_config.json", "pytorch_model.bin"]
    },
    {
        "id": "dmis-lab/biobert-base-cased-v1.2",
        "slug": "biobert",
        "files": ["config.json", "vocab.txt", "pytorch_model.bin"]
    },
    {
        "id": "emilyalsentzer/Bio_ClinicalBERT",
        "slug": "bioclinicalbert",
        "files": ["config.json", "vocab.txt", "pytorch_model.bin"]
    },
    {
        "id": "allenai/scibert_scivocab_uncased",
        "slug": "scibert",
        "files": ["config.json", "vocab.txt", "pytorch_model.bin"]
    }
]

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"  [OK] Already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return
        
    print(f"  Downloading {dest_path.name} ...", flush=True)
    temp_path = dest_path.with_suffix(".part")
    t0 = time.time()
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=90) as resp, open(temp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_log = t0
        while True:
            chunk = resp.read(1024 * 1024 * 4)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log > 4.0:
                pct = (downloaded / total * 100) if total else 0
                speed = (downloaded / (1024 * 1024)) / (now - t0 + 1e-5)
                print(f"    -> {downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                last_log = now
                
    temp_path.replace(dest_path)
    dur = time.time() - t0
    print(f"  [DONE] {dest_path.name}: {downloaded / (1024*1024):.1f} MB in {dur:.1f}s ({downloaded/(1024*1024)/dur:.2f} MB/s)")

def main():
    base_dir = Path(r"C:\Users\khan computer\.gemini\antigravity\scratch\trievo-clinical-ner\pretrained_backbones")
    for m in MODELS:
        dest_dir = base_dir / m["slug"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==========================================")
        print(f"Backbone: {m['id']} -> {m['slug']}")
        print(f"==========================================")
        base_url = f"https://huggingface.co/{m['id']}/resolve/main"
        for fname in m["files"]:
            url = f"{base_url}/{fname}"
            dest = dest_dir / fname
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")

if __name__ == "__main__":
    main()
