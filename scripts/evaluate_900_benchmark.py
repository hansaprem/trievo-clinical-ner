"""Strict evaluation of the 4 finalized models on their standardized 900-sentence test partitions."""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import prepare_hf_dataset

TEST_DIR = ROOT_DIR / "data" / "standardized_900_test"
REPORTS_DIR = ROOT_DIR / "reports"

BENCHMARK_CONFIGS = [
    {
        "name": "BC5CDR",
        "domain": "Biomedical Literature",
        "model_dir": ROOT_DIR / "models" / "bc5cdr_pubmedbert",
        "test_file": TEST_DIR / "bc5cdr_test_900.json",
        "entity_types": ["Chemical", "Disease"]
    },
    {
        "name": "NCBI Disease",
        "domain": "Biomedical Literature",
        "model_dir": ROOT_DIR / "models" / "ncbi_disease_pubmedbert",
        "test_file": TEST_DIR / "ncbi_disease_test_900.json",
        "entity_types": ["Disease"]
    },
    {
        "name": "JNLPBA",
        "domain": "Molecular Biology",
        "model_dir": ROOT_DIR / "models" / "jnlpba_pubmedbert",
        "test_file": TEST_DIR / "jnlpba_test_900.json",
        "entity_types": ["Protein", "DNA", "RNA", "Cell Type", "Cell Line"]
    },
    {
        "name": "AnatEM",
        "domain": "Biomedical Literature",
        "model_dir": ROOT_DIR / "models" / "anatem_pubmedbert",
        "test_file": TEST_DIR / "anatem_test_900.json",
        "entity_types": ["Anatomy"]
    }
]


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def evaluate_dataset(cfg: dict, tokenizer) -> dict:
    dataset_name = cfg["name"]
    model_dir = cfg["model_dir"]
    test_file = cfg["test_file"]

    print(f"\n=======================================================", flush=True)
    print(f"Evaluating {dataset_name} on Standardized 900-Sentence Test Set", flush=True)
    print(f"Model path: {model_dir}", flush=True)
    print(f"Test file:  {test_file}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Load label map
    with open(model_dir / "label.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}

    # 2. Load model
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    # 3. Load 900 test records
    test_records = load_jsonl(test_file)
    assert len(test_records) == 900, f"Expected 900 sentences, got {len(test_records)} in {test_file}"
    num_sentences = len(test_records)
    num_tokens = sum(len(r["tokens"]) for r in test_records)

    # 4. Tokenize & prepare dataset
    test_dataset = prepare_hf_dataset(test_records, tokenizer, label2id, max_length=128)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=str(ROOT_DIR / "tmp_eval_output"),
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )

    # 5. Run inference
    predictions_output = trainer.predict(test_dataset)
    preds_logits = predictions_output.predictions
    labels = predictions_output.label_ids

    preds = np.argmax(preds_logits, axis=2)

    true_predictions = [
        [id2label[p_idx] for (p_idx, l_idx) in zip(pred_row, label_row) if l_idx != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l_idx] for (p_idx, l_idx) in zip(pred_row, label_row) if l_idx != -100]
        for pred_row, label_row in zip(preds, labels)
    ]

    # Calculate strict entity metrics
    prec = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
    rec = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
    f1 = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)

    # Count gold and predicted entity spans
    gold_entities = sum(1 for sent in true_labels for tag in sent if tag.startswith("B-"))
    pred_entities = sum(1 for sent in true_predictions for tag in sent if tag.startswith("B-"))

    # Per-entity breakdown
    report_dict = classification_report(
        true_labels,
        true_predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0
    )

    per_entity = {}
    for ent_type, vals in report_dict.items():
        if isinstance(vals, dict) and "f1-score" in vals and ent_type not in ["micro avg", "macro avg", "weighted avg"]:
            per_entity[ent_type] = {
                "precision": float(vals["precision"]),
                "recall": float(vals["recall"]),
                "f1": float(vals["f1-score"]),
                "support": int(vals["support"])
            }

    print(f"Results for {dataset_name}:")
    print(f"  Test Sentences:     {num_sentences}")
    print(f"  Test Tokens:        {num_tokens:,}")
    print(f"  Gold Entities:      {gold_entities:,}")
    print(f"  Predicted Entities: {pred_entities:,}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  Entity Micro F1:    {f1:.4f}")
    print(f"  Per-Entity F1:")
    for k, v in per_entity.items():
        print(f"    - {k}: F1={v['f1']:.4f} (P={v['precision']:.4f}, R={v['recall']:.4f}, Support={v['support']})")

    return {
        "dataset": dataset_name,
        "domain": cfg["domain"],
        "entity_types": ", ".join(cfg["entity_types"]),
        "num_test_sentences": num_sentences,
        "num_test_tokens": num_tokens,
        "gold_entities": gold_entities,
        "pred_entities": pred_entities,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "per_entity": per_entity,
        "checkpoint_path": str(model_dir.relative_to(ROOT_DIR))
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT_DIR / "pretrained_backbones" / "pubmedbert"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    results = []
    for cfg in BENCHMARK_CONFIGS:
        res = evaluate_dataset(cfg, tokenizer)
        results.append(res)

    # Compute Macro-average across datasets
    macro_f1 = float(np.mean([r["f1"] for r in results]))
    macro_prec = float(np.mean([r["precision"] for r in results]))
    macro_rec = float(np.mean([r["recall"] for r in results]))

    print(f"\n=======================================================", flush=True)
    print(f"SUMMARY: Final Standardized 900-Sentence Benchmark", flush=True)
    print(f"Macro-average Precision across datasets: {macro_prec:.4f}", flush=True)
    print(f"Macro-average Recall across datasets:    {macro_rec:.4f}", flush=True)
    print(f"Macro-average F1 across datasets:        {macro_f1:.4f}", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Save final comparison CSV
    csv_path = REPORTS_DIR / "final_900_sentence_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset",
            "Domain",
            "Test_Sentences",
            "Test_Tokens",
            "Gold_Entities",
            "Predicted_Entities",
            "Precision",
            "Recall",
            "Entity_F1",
            "Entity_Types",
            "Checkpoint_Path"
        ])
        for r in results:
            writer.writerow([
                r["dataset"],
                r["domain"],
                r["num_test_sentences"],
                r["num_test_tokens"],
                r["gold_entities"],
                r["pred_entities"],
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                r["entity_types"],
                r["checkpoint_path"]
            ])
        writer.writerow([
            "Macro-average across datasets",
            "Multi-domain Average",
            "900 per dataset (3,600 total)",
            sum(r["num_test_tokens"] for r in results),
            sum(r["gold_entities"] for r in results),
            sum(r["pred_entities"] for r in results),
            f"{macro_prec:.4f}",
            f"{macro_rec:.4f}",
            f"{macro_f1:.4f}",
            "Chemical, Disease, Protein, DNA, RNA, Cell Type, Cell Line, Anatomy",
            "N/A"
        ])
    print(f"Saved comparison CSV to {csv_path}")

    # 2. Update Manifest with final evaluated metrics
    manifest_path = REPORTS_DIR / "final_900_sentence_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for r in results:
            d_name = r["dataset"]
            if d_name in manifest.get("datasets", {}):
                manifest["datasets"][d_name]["final_900_eval_metrics"] = {
                    "test_sentences": r["num_test_sentences"],
                    "test_tokens": r["num_test_tokens"],
                    "gold_entities": r["gold_entities"],
                    "pred_entities": r["pred_entities"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "per_entity": r["per_entity"]
                }
        manifest["macro_average_metrics_across_datasets"] = {
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Updated manifest with final evaluated metrics at {manifest_path}")

    # 3. Save Final Markdown Report
    report_md_path = REPORTS_DIR / "final_900_sentence_report.md"
    md_content = f"""# Final Standardized 900-Sentence Clinical & Biomedical NER Benchmark Report

**Date:** 2026-09-27  
**Project:** TriEvo Clinical NER (`trievo-clinical-ner`)  
**Lead AI Systems Engineer:** DeepMind Agentic Pair Programmer  
**Hardware Environment:** AMD64 Multi-Core CPU (`torch 2.11.0+cpu`, Windows 11)  
**Standardized Test Protocol:** Exactly **900 held-out test sentences per dataset** sampled with deterministic seed = 42.  
**Evaluation Standard:** Strict Entity-Level `seqeval` (IOB2 standard).  

---

## 1. Executive Summary & Final Benchmark Table

The **TriEvo Clinical NER** final benchmark provides a standardized, publication-grade evaluation of **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased`) across four foundational biomedical and clinical entity domains:
1. **BC5CDR:** Pharmacology & Pathology (`Chemical`, `Disease`)
2. **NCBI Disease:** Human Disease Genetics (`Disease`)
3. **JNLPBA (BioNLP 2004):** Molecular Biology (`Protein`, `DNA`, `RNA`, `Cell Type`, `Cell Line`)
4. **AnatEM:** Human Anatomy & Gross Pathological Sites (`Anatomy`)

### Final Standardized Benchmark Comparison Table

| Dataset | Test Sentences | Precision | Recall | Entity F1 | Entity Types |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BC5CDR** | **900** | **{results[0]['precision']:.4f}** | **{results[0]['recall']:.4f}** | **{results[0]['f1']:.4f}** | Chemical, Disease |
| **NCBI Disease** | **900** | **{results[1]['precision']:.4f}** | **{results[1]['recall']:.4f}** | **{results[1]['f1']:.4f}** | Disease |
| **JNLPBA** | **900** | **{results[2]['precision']:.4f}** | **{results[2]['recall']:.4f}** | **{results[2]['f1']:.4f}** | Protein, DNA, RNA, Cell Type, Cell Line |
| **AnatEM** | **900** | **{results[3]['precision']:.4f}** | **{results[3]['recall']:.4f}** | **{results[3]['f1']:.4f}** | Anatomy |

**Macro-average F1 across datasets:** **{macro_f1:.4f}**  
*(Macro-average Precision: {macro_prec:.4f} | Macro-average Recall: {macro_rec:.4f})*

> [!NOTE]
> **Evaluation Metric Policy:**  
> *Accuracy was not reported because entity-level F1 is the primary NER evaluation metric.* Background tokens (`O`) comprise ~90% of tokens in biomedical literature, which artificially inflates token-level accuracy while failing to measure boundary recognition. F1 is strictly calculated at the entity span level.

---

## 2. Per-Entity Metric Breakdown (Standardized 900 Test Sentences)

| Dataset | Entity Class | Precision | Recall | Entity F1 | Support (Gold Entities) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""

    for r in results:
        for ent_name, m in r["per_entity"].items():
            if m["support"] > 0:
                md_content += f"| **{r['dataset']}** | `{ent_name}` | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | {m['support']:,} |\n"

    md_content += f"""
---

## 3. Distinction from Previous Experiment Results

Previous exploratory experiments used disparate testing protocols, non-uniform test sizes, and differing sample budgets:
* **Previous BC5CDR:** Test F1 = 82.03% (evaluated on 200 held-out sentences)
* **Previous NCBI Disease:** Test F1 = 75.22% (evaluated on all 940 sentences)
* **Previous JNLPBA:** Test F1 = 70.01% (evaluated on all 3,856 sentences)
* **Archived MTSamples:** Test F1 = 64.56% (evaluated on 201 sentences, archived in `archive/legacy_mtsamples/`)

The **Final Standardized Benchmark** reported in Section 1 replaces variable-size testing with a rigorous, uniform **900 test sentences per dataset** protocol. All models were evaluated strictly against these frozen 900-sentence partitions. All previous experiment artifacts remain preserved on disk for complete auditability.

---

## 4. Test Partition Details & Token Counts

| Dataset | Test Sentences | Total Tokens | Gold Entities | Predicted Entities | Checkpoint Location |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BC5CDR** | 900 | {results[0]['num_test_tokens']:,} | {results[0]['gold_entities']:,} | {results[0]['pred_entities']:,} | [`models/bc5cdr_pubmedbert`](file:///{str(ROOT_DIR / 'models' / 'bc5cdr_pubmedbert').replace('\\', '/')}) |
| **NCBI Disease** | 900 | {results[1]['num_test_tokens']:,} | {results[1]['gold_entities']:,} | {results[1]['pred_entities']:,} | [`models/ncbi_disease_pubmedbert`](file:///{str(ROOT_DIR / 'models' / 'ncbi_disease_pubmedbert').replace('\\', '/')}) |
| **JNLPBA** | 900 | {results[2]['num_test_tokens']:,} | {results[2]['gold_entities']:,} | {results[2]['pred_entities']:,} | [`models/jnlpba_pubmedbert`](file:///{str(ROOT_DIR / 'models' / 'jnlpba_pubmedbert').replace('\\', '/')}) |
| **AnatEM** | 900 | {results[3]['num_test_tokens']:,} | {results[3]['gold_entities']:,} | {results[3]['pred_entities']:,} | [`models/anatem_pubmedbert`](file:///{str(ROOT_DIR / 'models' / 'anatem_pubmedbert').replace('\\', '/')}) |
| **Total Benchmark** | **3,600** | **{sum(r['num_test_tokens'] for r in results):,}** | **{sum(r['gold_entities'] for r in results):,}** | **{sum(r['pred_entities'] for r in results):,}** | **4 Distinct Dedicated Model Heads** |

---

## 5. Non-Clinical Regulatory Notice

> [!CAUTION]
> **Research Prototype Only:** The models, checkpoints, and evaluations documented herein were developed solely for academic NLP research and algorithmic comparison. None of these models are approved or certified as medical devices, diagnostic aids, or clinical decision support software by the FDA, EMA, or any national health regulatory authority.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved final report markdown to {report_md_path}")


if __name__ == "__main__":
    main()
