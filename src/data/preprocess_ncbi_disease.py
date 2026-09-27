"""Preprocessing and validation pipeline for the NCBI Disease Corpus."""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "ncbi_disease" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "ncbi_disease" / "processed"

LABEL2ID = {
    "O": 0,
    "B-Disease": 1,
    "I-Disease": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_conll_file(file_path: Path) -> Tuple[List[Dict], Dict]:
    """Parses a CoNLL TSV file into standardized records and collects statistics."""
    records = []
    current_tokens = []
    current_tags = []
    
    tag_counter = Counter()
    malformed_lines = 0
    malformed_sentences = 0
    invalid_transitions = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                if current_tokens:
                    # Validate sentence
                    if len(current_tokens) == 0:
                        malformed_sentences += 1
                    else:
                        # Check BIO transitions
                        for i, tag in enumerate(current_tags):
                            if tag == "I-Disease":
                                if i == 0 or current_tags[i - 1] == "O":
                                    invalid_transitions += 1
                        
                        tag_ids = [LABEL2ID[t] for t in current_tags]
                        records.append({
                            "tokens": current_tokens,
                            "tags": tag_ids
                        })
                    current_tokens = []
                    current_tags = []
            else:
                parts = line_str.split("\t")
                if len(parts) >= 2:
                    token, tag = parts[0], parts[1]
                elif len(parts) == 1:
                    token, tag = parts[0], "O"
                else:
                    malformed_lines += 1
                    continue
                    
                if tag not in LABEL2ID:
                    # Sanitize or flag unexpected tags
                    tag = "O"
                    malformed_lines += 1
                    
                tag_counter[tag] += 1
                current_tokens.append(token)
                current_tags.append(tag)
                
        # Trailing sentence if any
        if current_tokens:
            tag_ids = [LABEL2ID[t] for t in current_tags]
            records.append({
                "tokens": current_tokens,
                "tags": tag_ids
            })
            
    stats = {
        "sentences": len(records),
        "tokens": sum(len(r["tokens"]) for r in records),
        "disease_entities": tag_counter.get("B-Disease", 0),
        "tag_distribution": dict(tag_counter),
        "malformed_lines": malformed_lines,
        "malformed_sentences": malformed_sentences,
        "invalid_transitions": invalid_transitions
    }
    
    return records, stats


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    split_files = {
        "train": RAW_DIR / "train.tsv",
        "valid": RAW_DIR / "devel.tsv",
        "test": RAW_DIR / "test.tsv"
    }
    
    overall_stats = {}
    
    for split_name, raw_path in split_files.items():
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")
            
        records, stats = parse_conll_file(raw_path)
        overall_stats[split_name] = stats
        
        # Save processed jsonlines
        out_path = PROCESSED_DIR / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
                
        print(f"Processed [{split_name}]: {stats['sentences']} sentences, {stats['tokens']} tokens, {stats['disease_entities']} disease entities.")
        
    # Save label.json
    with open(PROCESSED_DIR / "label.json", "w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, indent=2)
        
    # Save statistics report
    stats_out = ROOT_DIR / "reports" / "ncbi_disease_statistics.json"
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2)
        
    print("\nNCBI Disease processing completed successfully.")
    print(f"Label map: {LABEL2ID}")
    print(f"Stats saved to {stats_out}")


if __name__ == "__main__":
    main()
