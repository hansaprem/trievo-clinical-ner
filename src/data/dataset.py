"""Data loading, preprocessing, and token-label alignment for Clinical NER."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import Dataset


def load_raw_dataset(data_dir: str) -> Tuple[Dict[str, List[dict]], dict, dict]:
    """Loads raw json lines splits and label mapping."""
    data_path = Path(data_dir)
    
    with open(data_path / "label.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    
    splits = {}
    for s in ["train", "valid", "test"]:
        filepath = data_path / f"{s}.json"
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        splits[s] = records
        
    return splits, label2id, id2label


def prepare_hf_dataset(records: List[dict], tokenizer, label2id: dict, max_length: int = 128) -> Dataset:
    """
    Prepares a tokenized Hugging Face Dataset with aligned label ids.
    Uses padding=False so DataCollatorForTokenClassification dynamically pads each batch,
    reducing unnecessary computation on padding tokens by up to 5x.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for r in records:
        tokens = r["tokens"]
        tags = r["tags"]
        
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        word_ids = tokenized.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx] if word_idx < len(tags) else -100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        all_input_ids.append(tokenized["input_ids"])
        all_attention_mask.append(tokenized["attention_mask"])
        all_labels.append(label_ids)
        
    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels
    })
