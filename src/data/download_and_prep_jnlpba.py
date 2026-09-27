"""Acquisition, standardization, and validation pipeline for JNLPBA (BioNLP 2004)."""

import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "jnlpba" / "processed"
RAW_DIR = ROOT_DIR / "data" / "jnlpba" / "raw"
REPORTS_DIR = ROOT_DIR / "reports"

BASE_URL = "https://huggingface.co/datasets/tner/bionlp2004/raw/main/dataset"

FILES = {
    "label.json": f"{BASE_URL}/label.json",
    "train.json": f"{BASE_URL}/train.json",
    "valid.json": f"{BASE_URL}/valid.json",
    "test.json": f"{BASE_URL}/test.json"
}


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} to {dest}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download raw files
    for filename, url in FILES.items():
        dest = RAW_DIR / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"Already exists: {dest}")

    # 2. Load and inspect label map
    with open(RAW_DIR / "label.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    print("\nJNLPBA Label Map:")
    print(json.dumps(label2id, indent=2))

    # Copy label.json to processed
    with open(PROCESSED_DIR / "label.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    stats = {
        "dataset": "JNLPBA / BioNLP 2004",
        "domain": "Molecular Biology (MEDLINE abstracts)",
        "entity_types": ["DNA", "RNA", "protein", "cell_type", "cell_line"],
        "num_classes": len(label2id),
        "splits": {},
        "tag_counts": {},
        "entity_counts": {},
        "integrity_checks": {
            "invalid_bio_transitions": 0,
            "empty_sentences": 0
        }
    }

    # 3. Process and validate each split
    for split in ["train", "valid", "test"]:
        src_path = RAW_DIR / f"{split}.json"
        dst_path = PROCESSED_DIR / f"{split}.json"

        tag_counter = Counter()
        entity_counter = Counter()
        valid_records = []
        token_count = 0

        with open(src_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                record = json.loads(line_str)
                tokens = record["tokens"]
                tags = record["tags"]

                if len(tokens) == 0:
                    stats["integrity_checks"]["empty_sentences"] += 1
                    continue

                assert len(tokens) == len(tags), f"Mismatch in {split} line {line_idx}"

                token_count += len(tokens)

                # Validate BIO transitions
                for i, tag_id in enumerate(tags):
                    tag_name = id2label[tag_id]
                    tag_counter[tag_name] += 1
                    if tag_name.startswith("I-"):
                        if i == 0 or id2label[tags[i - 1]] not in [tag_name, tag_name.replace("I-", "B-")]:
                            stats["integrity_checks"]["invalid_bio_transitions"] += 1
                            # Fix transition to B-
                            tags[i] = label2id[tag_name.replace("I-", "B-")]

                    if tag_name.startswith("B-"):
                        ent_type = tag_name[2:]
                        entity_counter[ent_type] += 1

                valid_records.append({
                    "tokens": tokens,
                    "tags": tags
                })

        # Write to processed
        with open(dst_path, "w", encoding="utf-8") as f:
            for rec in valid_records:
                f.write(json.dumps(rec) + "\n")

        stats["splits"][split] = {
            "sentences": len(valid_records),
            "tokens": token_count,
            "entities": sum(entity_counter.values())
        }
        stats["tag_counts"][split] = dict(tag_counter)
        stats["entity_counts"][split] = dict(entity_counter)

        print(f"Processed {split}: {len(valid_records):,} sentences, {token_count:,} tokens, {sum(entity_counter.values()):,} entities")
        print(f"  Entity breakdown: {dict(entity_counter)}")

    # 4. Save statistics report
    stats_file = REPORTS_DIR / "jnlpba_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nJNLPBA preprocessing complete. Statistics saved to {stats_file}")
    print(f"Total sentences across splits: {sum(s['sentences'] for s in stats['splits'].values()):,}")
    print(f"Total entities across splits: {sum(s['entities'] for s in stats['splits'].values()):,}")
    print(f"Invalid BIO transitions detected/repaired: {stats['integrity_checks']['invalid_bio_transitions']}")


if __name__ == "__main__":
    main()
