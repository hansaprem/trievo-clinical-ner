"""Test script to verify PubTator character offset indexing."""

import gzip
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
med_dir = root_dir / "data" / "medmentions" / "raw"

with gzip.open(med_dir / "corpus_pubtator.txt.gz", "rt", encoding="utf-8") as f:
    title, abstract = None, None
    tested = 0
    for line in f:
        line = line.strip()
        if not line:
            if tested > 0:
                break
            continue
        if "|t|" in line:
            title = line.split("|", 2)[2]
        elif "|a|" in line:
            abstract = line.split("|", 2)[2]
        else:
            parts = line.split("\t")
            start, end, text = int(parts[1]), int(parts[2]), parts[3]
            full_doc = title + " " + abstract
            slice_full = full_doc[start:end]
            match_full = (slice_full == text)
            print(f"Mention: '{text}' ({start}:{end}) | match in (title + ' ' + abstract): {match_full} (slice='{slice_full}')")
            tested += 1
            if tested >= 5:
                break
