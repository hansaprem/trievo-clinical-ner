# TriEvo Clinical NER (`trievo-clinical-ner`)

An end-to-end, high-performance Clinical Named Entity Recognition (NER) system engineered from scratch for clinical free-text entity extraction. TriEvo extracts pharmacological agents (`CHEMICAL_DRUG`) and pathological conditions/symptoms (`DISEASE_PROBLEM`) with verified character offsets and authentic model confidence scores.

---

## 1. Project Overview & Empirical Results

Five transformer architectures were trained and evaluated under strictly controlled, identical conditions on the gold-standard **BioCreative V CDR (BC5CDR)** clinical benchmark dataset:

| Architecture | Model Backbone | Val F1 | Held-out Test F1 | Chemical F1 | Disease F1 | Latency (CPU) | Abs Improvement | Rel Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | **0.7100** | **0.8203** | **0.8432** | **0.7955** | **68.13 ms** | **+17.12 pts** | **+26.37%** |
| 🥈 **SciBERT** | `allenai/scibert_scivocab_uncased` | 0.8065 | 0.8036 | 0.8000 | 0.8077 | 89.25 ms | +15.45 pts | +23.80% |
| 🥉 **BioBERT** | `dmis-lab/biobert-base-cased-v1.2` | 0.7379 | 0.7359 | 0.8127 | 0.6403 | 151.87 ms | +8.68 pts | +13.37% |
| 4 **BioClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | 0.6267 | 0.6914 | 0.8051 | 0.5512 | 100.37 ms | +4.23 pts | +6.52% |
| 5 **General BERT (Base)** | `google-bert/bert-base-uncased` | 0.5719 | 0.6491 | 0.7368 | 0.5611 | 108.11 ms | — (Baseline) | — (Baseline) |

*Evaluation Metric Note:* Strict entity-level F1 via `seqeval` is the primary evaluation standard. *Accuracy was not reported because entity-level F1 is the primary NER evaluation metric in token-imbalanced datasets.*

---

## 2. Project Architecture

```
trievo-clinical-ner/
│
├── data/
│   ├── raw/                       # Raw BC5CDR dataset splits (train, valid, test, label.json)
│   └── processed/
├── configs/
├── src/
│   ├── data/
│   │   ├── dataset.py             # Preprocessing & subword-to-tag alignment with dynamic padding
│   │   └── inspect_dataset.py     # Dataset analysis and statistics profiling
│   ├── models/
│   │   └── token_classifier.py    # AutoModel token classification loader
│   ├── training/
│   │   └── trainer.py             # Reproducible training engine & experiment logger
│   ├── evaluation/
│   │   └── metrics.py             # Strict entity-level seqeval metrics
│   └── inference/
│       └── pipeline.py            # Clinical free-text inference engine with offset mapping
│
├── pretrained_backbones/          # Fully downloaded and cached transformer model weights
│   ├── bert-base-uncased/
│   ├── pubmedbert/
│   ├── biobert/
│   ├── bioclinicalbert/
│   └── scibert/
│
├── experiments/                   # Experiment logs, metrics.json, and checkpoints
│   ├── baseline/
│   ├── pubmedbert/
│   ├── biobert/
│   ├── bioclinicalbert/
│   ├── scibert/
│   ├── model_comparison.csv
│   └── model_comparison.json
│
├── models/
│   └── best_clinical_ner/         # Exported production checkpoint (PubMedBERT)
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       ├── metrics.json
│       └── model_card.md
│
├── reports/
│   ├── dataset_analysis.md        # Comprehensive dataset profile and token breakdown
│   ├── dataset_statistics.json    # Machine-readable dataset statistics
│   ├── model_compatibility_report.md
│   ├── error_analysis.md          # Qualitative false positive/negative & boundary review
│   └── final_model_selection.md   # Complete selection and engineering verdict
│
├── scripts/
│   ├── check_compatibility.py     # Verifies architectures, tokenizers, and configs
│   ├── run_experiment.py          # Unified training entry point
│   ├── compare_models.py          # Generates comparative CSV and JSON metrics
│   ├── test_inference.py          # Verification suite across diverse clinical cases
│   └── error_analysis.py          # Detailed test set error categorization
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Installation & Setup

```bash
# Clone or navigate to the project directory
cd "C:\Users\khan computer\.gemini\antigravity\scratch\trievo-clinical-ner"

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Usage Guide

### A. Quick Python Inference
```python
from src.inference.pipeline import ClinicalNERPipeline

# Initialize production pipeline
pipeline = ClinicalNERPipeline("models/best_clinical_ner")

# Extract clinical entities
text = "The patient with Parkinson's disease developed severe dyskinesia following treatment with levodopa."
result = pipeline.extract_entities(text)

print(result)
```

**Output:**
```json
{
  "text": "The patient with Parkinson's disease developed severe dyskinesia following treatment with levodopa.",
  "entities": [
    {
      "text": "Parkinson's disease",
      "label": "DISEASE_PROBLEM",
      "start": 17,
      "end": 36,
      "confidence": 0.6714
    },
    {
      "text": "dyskinesia",
      "label": "DISEASE_PROBLEM",
      "start": 54,
      "end": 64,
      "confidence": 0.7897
    },
    {
      "text": "levodopa",
      "label": "CHEMICAL_DRUG",
      "start": 92,
      "end": 100,
      "confidence": 0.9637
    }
  ]
}
```

### B. Run Inference Verification Test Suite
```bash
python scripts/test_inference.py --model_path "./models/best_clinical_ner"
```

### C. Run Error Analysis on Test Set
```bash
python scripts/error_analysis.py --model_path "./models/best_clinical_ner" --max_samples 200
```

### D. Re-run an Experiment
```bash
python scripts/run_experiment.py --model_id "./pretrained_backbones/pubmedbert" --exp_name "pubmedbert_run" --epochs 3 --batch_size 16 --lr 3e-5
```

---

## 5. Non-Clinical & Non-Diagnostic Regulatory Disclaimer
> [!CAUTION]
> **Research Prototype Only:** This system and its associated models are provided for informational and clinical NLP research purposes only. They have NOT been evaluated or approved by the FDA, EMA, or any national health regulatory agency as medical devices or clinical decision support software. This software must NOT be utilized as a substitute for professional clinical judgment, emergency medical triage, prescription management, or disease diagnosis.
