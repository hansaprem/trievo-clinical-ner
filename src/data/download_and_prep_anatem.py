"""Acquisition, standardization, and validation pipeline for AnatEM (Anatomical Entity Mention Corpus)."""

import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "anatem" / "processed"
RAW_DIR = ROOT_DIR / "data" / "anatem" / "raw"
REPORTS_DIR = ROOT_DIR / "reports"

BASE_URL = "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/AnatEM-IOB"

FILES = {
    "train.tsv": f"{BASE_URL}/train.tsv",
    "devel.tsv": f"{BASE_URL}/devel.tsv",
    "test.tsv": f"{BASE_URL}/test.tsv"
}

LABEL2ID = {
    "O": 0,
    "B-Anatomy": 1,
    "I-Anatomy": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} to {dest}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def parse_tsv_to_jsonl(tsv_path: Path) -> list:
    """Parses standard CoNLL TSV (<token>\t<tag>) into list of {'tokens': [...], 'tags': [...]} records."""
    records = []
    current_tokens = []
    current_tags = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                if current_tokens:
                    records.append({
                        "tokens": current_tokens,
                        "tags": current_tags
                    })
                    current_tokens = []
                    current_tags = []
            else:
                parts = line_str.split("\t")
                if len(parts) >= 2:
                    token, tag = parts[0], parts[-1]
                else:
                    parts = line_str.split()
                    token, tag = parts[0], parts[-1]

                # Map tag to ID
                if tag not in LABEL2ID:
                    # In case of minor variations
                    if "Anatomy" in tag:
                        tag = "B-Anatomy" if tag.startswith("B-") else "I-Anatomy"
                    else:
                        tag = "O"

                tag_id = LABEL2ID[tag]
                current_tokens.append(token)
                current_tags.append(tag_id)

    if current_tokens:
        records.append({
            "tokens": current_tokens,
            "tags": current_tags
        })

    return records


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

    # Save label.json
    with open(PROCESSED_DIR / "label.json", "w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, indent=2)

    stats = {
        "dataset": "AnatEM (Anatomical Entity Mention Corpus)",
        "source": "Pyysalo & Ananiadou (2014) / Cambridge MTL-Bioinformatics",
        "domain": "Biomedical Literature (PubMed Abstracts)",
        "entity_types": ["Anatomy"],
        "label_map": LABEL2ID,
        "splits": {},
        "tag_counts": {},
        "entity_counts": {},
        "integrity_checks": {
            "invalid_bio_transitions": 0,
            "empty_sentences": 0
        }
    }

    split_map = {
        "train": RAW_DIR / "train.tsv",
        "valid": RAW_DIR / "devel.tsv",
        "test": RAW_DIR / "test.tsv"
    }

    # 2. Process, validate BIO transitions, and save
    for split_name, tsv_file in split_map.items():
        raw_records = parse_tsv_to_jsonl(tsv_file)
        valid_records = []
        token_count = 0
        tag_counter = Counter()
        entity_count = 0

        for rec in raw_records:
            tokens = rec["tokens"]
            tags = rec["tags"]

            if not tokens:
                stats["integrity_checks"]["empty_sentences"] += 1
                continue

            token_count += len(tokens)

            # Validate and repair any invalid BIO transitions
            for i, tag_id in enumerate(tags):
                tag_name = ID2LABEL[tag_id]
                tag_counter[tag_name] += 1

                if tag_name.startswith("I-"):
                    if i == 0 or ID2LABEL[tags[i - 1]] not in [tag_name, tag_name.replace("I-", "B-")]:
                        stats["integrity_checks"]["invalid_bio_transitions"] += 1
                        tags[i] = LABEL2ID[tag_name.replace("I-", "B-")]

                if tag_name.startswith("B-"):
                    entity_count += 1

            valid_records.append({
                "tokens": tokens,
                "tags": tags
            })

        # Save to processed JSONL
        out_jsonl = PROCESSED_DIR / f"{split_name}.json"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in valid_records:
                f.write(json.dumps(rec) + "\n")

        stats["splits"][split_name] = {
            "sentences": len(valid_records),
            "tokens": token_count,
            "entities": entity_count
        }
        stats["tag_counts"][split_name] = dict(tag_counter)
        stats["entity_counts"][split_name] = {"Anatomy": entity_count}

        print(f"Processed AnatEM {split_name}: {len(valid_records):,} sentences, {token_count:,} tokens, {entity_count:,} entities")
        print(f"  Tag counts: {dict(tag_counter)}")

    # 3. Save statistics report
    stats_file = REPORTS_DIR / "anatem_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAnatEM preprocessing complete. Statistics saved to {stats_file}")
    print(f"Total sentences across splits: {sum(s['sentences'] for s in stats['splits'].values()):,}")
    print(f"Total entities across splits: {sum(s['entities'] for s in stats['splits'].values()):,}")
    print(f"Invalid BIO transitions repaired: {stats['integrity_checks']['invalid_bio_transitions']}")


if __name__ == "__main__":
    main()
