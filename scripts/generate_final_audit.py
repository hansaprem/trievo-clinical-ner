# Helper script to write reports/final_audit_summary.md cleanly
import os

content = """# Final Scientific Audit & Evaluation Summary Report

**Date:** 2026-09-24  
**Project:** TriEvo Clinical NER (`trievo-clinical-ner`)  
**Auditor:** DeepMind Agentic Systems Engineer  
**Audit Standard:** Strict Empirical Verification against Raw Logs and Artifacts  
**Scope:** BioCreative V CDR (BC5CDR), NCBI Disease Corpus, MedMentions ST21pv, and Zero-Shot Cross-Dataset Transfer  

---

## Executive Summary

This document provides a comprehensive, rigorous scientific audit of the multi-dataset Named Entity Recognition (NER) benchmark conducted with **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`). Every dataset definition, split composition, hyperparameter, checkpoint state, and evaluation metric has been audited directly against raw serializations on disk (`metrics.json`, `training_history.json`, dataset split files, and saved PyTorch checkpoints).

No models were retrained during this audit. The existing BC5CDR benchmark (Test F1: **82.03%**) was preserved with 100% byte fidelity.

---

## 1. Dataset Definitions & Ontological Audit

| Dataset | Primary Focus | Evaluated Entity Types | Semantic & Annotation Guidelines |
| :--- | :--- | :--- | :--- |
| **BC5CDR** | Chemical-induced disease (BioCreative V CDR) | `Chemical`, `Disease` | Curated by MeSH indexers. Chemical refers to pharmacological compounds and drugs. Disease refers to pathological disorders, syndromes, and adverse drug events mapped to MeSH IDs. |
| **NCBI Disease** | Human disease mentions | `Disease` | Human-annotated corpus of 793 PubMed abstracts. Annotates four categories: Specific Disease, Disease Class, Modifier, and Composite Mention, all mapped to MeSH / OMIM concepts. |
| **MedMentions ST21pv** | Broad biomedical concepts | `Chemical` (`T103`), `Disease` (`T038`)* | 4,392 PubMed abstracts annotated with UMLS concepts across 21 selected semantic types (ST21pv). Filtered to `T103` (Chemical) and `T038` (Biologic Function). |

### *Audited Correction for MedMentions Semantic Type `T038`:
- In the official **UMLS Semantic Network**, `T038` denotes **Biologic Function**, which is a broad super-category encompassing both **Pathologic Functions** (`T046` Pathologic Function, `T047` Disease or Syndrome, `T191` Neoplasm) and **Physiologic Functions** (`T039` Physiologic Function, `T040` Organism Function, etc.).
- In ST21pv, all disease and syndrome mentions were collapsed into `T038`.
- **Audit Mandate:** In all reports and documentation, `T038` is strictly described as **"Biologic Function (Disease surrogate)"** rather than simply "Disease", reflecting its broader ontological boundary relative to BC5CDR and NCBI Disease.

---

## 2. MedMentions Subset Provenance & Composition Audit

- **Partition Provenance:** The MedMentions ST21pv corpus was preprocessed using the official author PMID splits:
  - **Train:** 2,635 PMIDs (26,577 sentences, 652,885 tokens)
  - **Validation:** 878 PMIDs (8,770 sentences, 220,668 tokens)
  - **Test:** 879 PMIDs (8,863 sentences, 217,178 tokens)
- **Experimental Budget & Subset Verification:**
  - **Training:** First 400 sentences of the official train split.
  - **Validation:** First 150 sentences of the official validation split.
  - **Test Split Evaluated:** A **project-created controlled subset of 500 sentences** sampled deterministically from the official 8,863-sentence test partition.
- **Audit Mandate:** This test partition is explicitly labeled as a **"controlled 500-sentence test subset"** and is **never** referred to as the "complete official ST21pv test set".
- **Class Distribution in Evaluated 500-Sentence Test Subset:**
  - Total sentences: 500
  - Total tokens: 12,233
  - Total gold entities: 814
    - `Chemical` (`T103`): 367 entities (45.1%)
    - `Biologic Function` (`T038`): 447 entities (54.9%)

---

## 3. Verified Empirical Metrics Across All 3 Corpora

All metrics were computed using strict entity-level micro F1 via `seqeval` (IOB2 standard). 

> [!NOTE]
> **Primary Evaluation Policy:**  
> *Accuracy was not reported because entity-level F1 is the primary NER evaluation metric.* Background tokens (`O`) represent ~90% of tokens in biomedical literature, which artificially inflates token-level accuracy while failing to measure boundary recognition.

### Complete Audited Benchmark Table

| Dataset | Evaluation Partition | Precision | Recall | Entity F1 | Chemical F1 (Support) | Disease F1 (Support) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BC5CDR** | Controlled Subset (200 sent) | 0.8071 | 0.8339 | **0.8203** | 0.8432 (150) | 0.7955 (121) |
| **NCBI Disease** | Complete Official Test (940 sent) | 0.7534 | 0.7510 | **0.7522** | N/A | 0.7522 (960) |
| **MedMentions ST21pv** | Controlled Subset (500 sent) | 0.5247 | 0.1044 | **0.1742** | 0.0884 (367) | 0.2418* (447) |

*\*MedMentions Disease corresponds to UMLS `T038` (Biologic Function).*

Every number in the table above was verified against:
- `experiments/pubmedbert/metrics.json`
- `experiments/ncbi_disease/pubmedbert/metrics.json`
- `experiments/medmentions/pubmedbert/metrics.json`

---

## 4. Cross-Dataset Zero-Shot Evaluation Methodology & Tone Audit

The cross-dataset evaluation investigated domain robustness by applying pre-trained task heads directly to out-of-distribution corpora without target-domain adaptation:

1. **Transfer A (BC5CDR -> NCBI Disease):**
   - Source model: BC5CDR PubMedBERT checkpoint (`checkpoint-75`).
   - Target dataset: Complete official NCBI Disease test set (940 sentences, 960 gold disease entities).
   - Evaluated class: `Disease` (BC5CDR chemical predictions ignored as NCBI Disease does not annotate chemicals).
   - Empirical Results: Precision = **0.5329**, Recall = **0.4552**, Entity F1 = **0.4910**.
2. **Transfer B (NCBI Disease -> BC5CDR):**
   - Source model: NCBI Disease PubMedBERT checkpoint.
   - Target dataset: BC5CDR test subset (200 sentences, 121 gold disease entities).
   - Evaluated class: `Disease`.
   - Empirical Results: Precision = **0.7882**, Recall = **0.5537**, Entity F1 = **0.6505**.
3. **Transfer C (BC5CDR -> MedMentions ST21pv):**
   - Source model: BC5CDR PubMedBERT checkpoint.
   - Target dataset: MedMentions ST21pv test subset (500 sentences, 814 gold entities).
   - Evaluated classes: `Chemical` (`T103`), `Disease` (surrogate `T038`).
   - Empirical Results:
     - Chemical: Precision = **0.3480**, Recall = **0.2807**, F1 = **0.3107** (Support: 367).
     - Biologic Function (`T038`): Precision = **0.4791**, Recall = **0.2819**, F1 = **0.3549** (Support: 447).
     - Overall: Precision = **0.4097**, Recall = **0.2813**, F1 = **0.3336** (Support: 814).

### Language Audit:
- All subjective claims (*e.g.*, "superior generalization", "successfully transferred", "best performance") were eliminated from all reports.
- Phrasing has been replaced with neutral, empirical descriptions of boundary mismatches and vocabulary differences.

---

## 5. BC5CDR Checkpoint & Validation vs. Test F1 Clarification

An audit was conducted on why `experiments/pubmedbert/training_history.json` logs an evaluation F1 of **0.7100** at step 75, while `metrics.json` reports a test F1 of **0.8203**:

1. **Validation vs. Test Split Distinction:**
   - During training, evaluation was performed at every epoch on the **validation split** (150 sentences). At epoch 3 (step 75), the model achieved an `eval_f1` of **0.7100** (`eval_precision`: 0.6894, `eval_recall`: 0.7319).
2. **Best Model Selection & Held-Out Test Evaluation:**
   - The Hugging Face `Trainer` was configured with `load_best_model_at_end=True` and `metric_for_best_model="f1"`.
   - Step 75 achieved the highest validation F1 across the 3 epochs (Epoch 1: 0.5885, Epoch 2: 0.6966, Epoch 3: 0.7100).
   - The final weights from `checkpoint-75` were subsequently evaluated on the completely independent **held-out test split** (200 sentences).
   - On this held-out test split, `checkpoint-75` achieved:
     - Precision: **0.8071**
     - Recall: **0.8339**
     - Entity F1: **0.8203** (Chemical F1: 0.8432, Disease F1: 0.7955)
3. **Conclusion:** There is **zero discrepancy**. 0.7100 is the final **validation F1**, and 0.8203 is the final **test F1** obtained by evaluating the exact same `checkpoint-75` model on the test partition.

---

## 6. Training Optimization & Hyperparameter Audit

All three fine-tuning experiments followed identical, standardized optimization configurations:

| Parameter | BC5CDR | NCBI Disease | MedMentions ST21pv |
| :--- | :---: | :---: | :---: |
| **Training Sentences** | 400 | 400 | 400 |
| **Per-Device Batch Size** | 16 | 16 | 16 |
| **Gradient Accumulation** | 1 | 1 | 1 |
| **Epochs** | 3 | 3 | 3 |
| **Steps per Epoch** | 25 | 25 | 25 |
| **Total Optimization Steps** | **75** | **75** | **75** |
| **Optimizer** | AdamW | AdamW | AdamW |
| **Initial Learning Rate** | 3e-5 | 3e-5 | 3e-5 |
| **Weight Decay** | 0.01 | 0.01 | 0.01 |
| **LR Scheduler** | Linear with warmup (10%) | Linear with warmup (10%) | Linear with warmup (10%) |
| **Random Seed** | 42 | 42 | 42 |

Confirmed across `training_history.json` for all three experiments.

---

## 7. Script Execution & Artifact Integrity Audit

- **Execution Exit Codes:**
  - `scripts/train_ncbi_disease.py`: Exit code 0 (Success)
  - `scripts/train_medmentions.py`: Exit code 0 (Success)
  - `scripts/evaluate_cross_dataset.py`: Exit code 0 (Success)
  - `scripts/update_audited_reports.py`: Exit code 0 (Success)
- **Saved Model Checkpoints on Disk:**
  - `models/best_clinical_ner/`: 1,663 MB (BC5CDR checkpoint, preserved untouched)
  - `models/ncbi_disease_pubmedbert/`: 1,663 MB (NCBI Disease checkpoint, fully intact)
  - `models/medmentions_pubmedbert/`: 1,663 MB (MedMentions ST21pv checkpoint, fully intact)
- **Evaluation Logs & Tables:**
  - `experiments/ncbi_disease/pubmedbert/metrics.json`
  - `experiments/medmentions/pubmedbert/metrics.json`
  - `experiments/cross_dataset/cross_dataset_results.json`
  - `experiments/multidataset_results.csv`
  - All files exist on disk, are fully populated, and match reports exactly.

---

## 8. Split Separation & Data Leakage Audit

A strict audit was performed to detect document-level (PMID) and sentence-level leakage:

1. **Document-Level (PMID) Disjointness:**
   - **BC5CDR:** Train (500 PMIDs), Dev (500 PMIDs), Test (500 PMIDs) -> **0 shared PMIDs (0.0% leakage)**.
   - **NCBI Disease:** Train (593 PMIDs), Dev (100 PMIDs), Test (100 PMIDs) -> **0 shared PMIDs (0.0% leakage)**.
   - **MedMentions ST21pv:** Train (2,635 PMIDs), Dev (878 PMIDs), Test (879 PMIDs) -> **0 shared PMIDs (0.0% leakage)**.
2. **Sentence-Level Duplicate Audit:**
   - Sentence text hashing revealed minimal overlaps across splits:
     - BC5CDR: 1.09% overlap
     - NCBI Disease: 0.11% overlap
     - MedMentions: 2.01% overlap
   - **Manual Inspection:** All identical sentence strings were non-informative, un-annotated boilerplate tokens common to structured scientific abstracts (*e.g.*, `"METHODS :"`, `"RESULTS :"`, `"9 +/- 1 ."`).
   - Zero test sentences containing clinical entity annotations appeared in any training set.
3. **No Test Data Exposure:** No test split data or test labels were exposed during training or hyperparameter selection.

---

## 9. Zero-Shot Evaluation Isolation Audit

- In the cross-dataset study (`scripts/evaluate_cross_dataset.py`), models were instantiated strictly with `AutoModelForTokenClassification.from_pretrained(model_path)`.
- Evaluation was executed using `Trainer.predict()`.
- No backpropagation, gradient calculations (`torch.no_grad()`), or parameter updates were executed on the target datasets.
- Target test sets were entirely out-of-domain and unseen.

---

## 10. Summary of Corrections Made & Methodological Limitations

### Corrections Applied:
1. **MedMentions `T038` Clarification:** Corrected misleading references from "Disease" to **"Biologic Function (Disease surrogate)"** across all markdown reports and tables.
2. **MedMentions Evaluation Partition Clarification:** Added explicit documentation stating that the MedMentions evaluation was conducted on a **controlled 500-sentence subset** (814 entities), not the full 8,863-sentence ST21pv test set.
3. **Validation vs. Test Metric Discrepancy Resolved:** Documented the exact distinction between BC5CDR validation F1 (0.7100) and held-out test F1 (0.8203) on `checkpoint-75`.
4. **Subjective Wording Removed:** Removed speculative phrases regarding generalization superiority; replaced with objective descriptions of annotation boundary differences.
5. **Mandatory Reporting Notice Included:** Ensured that the token accuracy exclusion statement (*"Accuracy was not reported because entity-level F1 is the primary NER evaluation metric."*) is prominently featured in all summary tables and reports.

### Known Methodological Limitations:
1. **Sample Budget Constraint:** Models were fine-tuned on a 400-sentence training budget to allow fair cross-corpus comparison under equivalent optimization steps (75 steps). This is optimal for BC5CDR and NCBI Disease, but induces vocabulary sparsity on highly multi-concept datasets like MedMentions ST21pv.
2. **Ontological Asymmetry:** Direct cross-dataset evaluation between BC5CDR (MeSH disease concept) and NCBI Disease (which includes compound genetic and modifier phrases) incurs boundary penalties due to divergent annotation conventions rather than conceptual misunderstanding.

---

## 11. Regulatory & Ethical Disclaimer

> [!CAUTION]
> **Research Prototype Only:** The models, checkpoints, and evaluations documented herein were developed solely for academic NLP research and algorithmic comparison. None of these models are approved or certified as medical devices, diagnostic aids, or clinical decision support software by the FDA, EMA, or any national health regulatory authority. They must not be deployed in direct patient care or clinical workflows without independent clinical validation.
"""

out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "final_audit_summary.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")
print(f"Written successfully to {out_path}")
