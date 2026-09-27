# Model Card: TriEvo PubMedBERT Clinical NER

## Model Overview
- **Model Name:** TriEvo PubMedBERT Clinical Named Entity Recognizer
- **Backbone Architecture:** PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)
- **Parameters:** 108,895,493 (108.9M)
- **Task:** Token Classification / Named Entity Recognition (NER)
- **Domain:** Biomedical & Clinical Pharmacology / Pathology
- **Tagging Format:** BIO (`O`, `B-Chemical`, `B-Disease`, `I-Disease`, `I-Chemical`)
- **Release Version:** 1.0.0
- **Release Date:** 2026-09-24

## Training Data & Provenance
- **Dataset:** BioCreative V CDR (Chemical-Disease Relation Task)
- **Source:** Peer-reviewed PubMed literature abstracts annotated by domain experts
- **Training Samples:** 400 sentences (isolated official training partition)
- **Validation Samples:** 150 sentences (official validation partition)
- **Held-Out Test Samples:** 200 sentences (official held-out test partition)
- **Class Distribution in Dataset:**
  - Chemical Entities: 5,203 train, 5,347 val, 5,385 test spans
  - Disease Entities: 4,182 train, 4,244 val, 4,424 test spans

## Evaluation Methodology & Performance
Evaluation strictly follows entity-level `seqeval` (IOB2 standard, exact boundary and class matching). Token accuracy is deliberately excluded as it is distorted by the 88%+ majority of non-entity `O` tokens.

| Metric | Result | vs Baseline (`bert-base-uncased`) |
| :--- | :---: | :---: |
| **Overall Test Entity F1** | **82.03%** | **+17.12 pts (+26.37% rel)** |
| **Overall Test Precision** | **80.71%** | +22.08 pts |
| **Overall Test Recall** | **83.39%** | +10.70 pts |
| **Chemical Entity F1** | **84.32%** | +10.64 pts |
| **Chemical Precision** | **88.32%** | +15.59 pts |
| **Chemical Recall** | **80.67%** | +6.00 pts |
| **Disease Entity F1** | **79.55%** | +23.44 pts |
| **Disease Precision** | **73.43%** | +26.73 pts |
| **Disease Recall** | **86.78%** | +16.53 pts |
| **Validation F1** | **71.00%** | +13.81 pts |
| **Inference Latency** | **68.13 ms / sentence** | -37.0% faster than baseline |

## Hyperparameters
- **Optimizer:** AdamW (`weight_decay=0.01`)
- **Learning Rate:** 3e-5 (linear warmup & decay)
- **Batch Size:** 16 (dynamic batch padding)
- **Epochs:** 3
- **Random Seed:** 42 (fixed across all candidate comparisons)
- **Max Sequence Length:** 128 subwords

## Intended Use
- Automated extraction of medications, drugs, chemical compounds, illnesses, phenotypes, and disease entities from clinical text.
- Research analysis of scientific abstracts, discharge summaries, and medical records.
- Input feature extraction for clinical triage, knowledge graph construction, and pharmacovigilance pipelines.

## Non-Clinical & Non-Diagnostic Disclaimer
> [!CAUTION]
> **Not a Medical Device:** This model is an artificial intelligence research prototype and is NOT certified as a diagnostic tool or clinical decision support device. It must NOT be used as a standalone system for patient diagnosis, treatment planning, prescription validation, or emergency triage without licensed medical professional supervision.

## Limitations
1. **Acronym Ambiguity:** Unconventional clinical shorthand may lead to false positives if the abbreviation matches common biological terms.
2. **Boundary Granularity:** Complex nested disease phrases (e.g. adjectival modifiers like "mild intermittent" vs "asthma") may show partial boundary shifts.
3. **Casing:** Built on an uncased backbone, so capitalized distinction in certain rare gene/drug acronyms relies on contextual rather than orthographic features.
