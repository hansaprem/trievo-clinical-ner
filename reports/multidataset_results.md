# Multi-Dataset Clinical & Biomedical NER Benchmark Results (Audited)

**Date:** 2026-09-24  
**Project:** TriEvo Clinical NER (`trievo-clinical-ner`)  
**Lead AI Systems Engineer:** DeepMind Agentic Pair Programmer  
**Hardware Environment:** AMD64 Multi-Core CPU (`torch 2.11.0`, Windows 11)  
**Evaluation Standard:** Strict Entity-Level `seqeval` (IOB2 standard)  
**Audit Status:** Verified against raw `metrics.json` and training logs.  

---

## 1. Executive Summary & Benchmark Table

This report presents the empirical evaluation of **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`) across three biomedical and clinical corpora:
1. **BioCreative V CDR (BC5CDR):** Dual Chemical and Disease extraction (*Existing Reference Benchmark — 100% Unchanged*).
2. **NCBI Disease Corpus:** Dedicated human disease mention extraction on the complete official test split.
3. **MedMentions (ST21pv):** Multi-concept biomedical entity recognition evaluated on a controlled 500-sentence test subset.

### Audited Benchmark Results Table

| Dataset | Model | Validation F1 | Test Precision | Test Recall | Test Entity F1 | Evaluation Scope | Evaluated Entities |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **BC5CDR** | PubMedBERT | 0.7100 | 0.8071 | 0.8339 | **0.8203** | Controlled Subset (200 sent) | 271 entities (150 Chem, 121 Dis) |
| **NCBI Disease** | PubMedBERT | 0.7509 | 0.7534 | 0.7510 | **0.7522** | Complete Official Test (940 sent) | 960 entities (All Disease) |
| **MedMentions ST21pv** | PubMedBERT | 0.2215 | 0.5247 | 0.1044 | **0.1742** | Controlled Subset (500 sent) | 814 entities (367 Chem, 447 Dis) |

> [!NOTE]
> **Primary Evaluation Policy:**  
> *Accuracy was not reported because entity-level F1 is the primary NER evaluation metric.* In clinical and biomedical NER, background tokens (`O`) comprise 88%–92% of all tokens, rendering token accuracy deceptively high (>95%) while failing to reflect true entity boundary identification.

---

## 2. Per-Entity Metric Breakdown

| Dataset | Entity Class | Target Concept | Precision | Recall | Entity F1 | Support (Test) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **BC5CDR** | Chemical | Pharmacological / Chemical compounds | 0.8832 | 0.8067 | **0.8432** | 150 |
| **BC5CDR** | Disease | MeSH Disease / Mental disorders | 0.7343 | 0.8678 | **0.7955** | 121 |
| **NCBI Disease** | Disease | Disease mentions (Specific, Class, Modifier) | 0.7534 | 0.7510 | **0.7522** | 960 |
| **MedMentions ST21pv** | Chemical (`T103`) | UMLS Chemical semantic type | 0.3016 | 0.0518 | **0.0884** | 367 |
| **MedMentions ST21pv** | Disease (`T038`)* | UMLS Biologic Function (Pathology / Disease) | 0.6667 | 0.1477 | **0.2418** | 447 |

*\*Note on MedMentions T038:* In the official UMLS Semantic Network, `T038` denotes **Biologic Function**, a broad super-category encompassing both Pathologic Functions (`T046`, `T047` Disease or Syndrome, `T191` Neoplasm) and normal Physiologic Functions (`T039`). In this study, `T038` was mapped as a surrogate for disease/pathological entities. It is broader in scope than the strict disease definitions in BC5CDR and NCBI Disease.

---

## 3. Training & Optimization Audit

- **Hyperparameters:** Standardized across all runs: AdamW optimizer, learning rate $3\times 10^{-5}$, weight decay $0.01$, batch size 16, dynamic batch padding, random seed 42.
- **Optimization Steps:** Exactly **75 steps** per experiment (400 training sentences / batch size 16 = 25 batches/epoch $\times$ 3 epochs = 75 steps).
- **Model Checkpoints:**
  - **BC5CDR:** `checkpoint-75` achieved validation F1 of 0.7100 during training; evaluating `checkpoint-75` on the held-out test split produced the final test F1 of 0.8203.
  - **NCBI Disease:** Best checkpoint saved at epoch 3 with validation F1 of 0.7509; test F1 was 0.7522.
  - **MedMentions ST21pv:** Best checkpoint saved at epoch 3 with validation F1 of 0.2215; test F1 was 0.1742.

---

## 4. Evaluation Scope & Sample Selection Methodology

1. **BC5CDR:**
   - Training: First 400 sentences of official train split (5,228 total).
   - Validation: First 150 sentences of official validation split (5,330 total).
   - Test: First 200 sentences of official held-out test split (5,865 total).
2. **NCBI Disease:**
   - Training: First 400 sentences of official train split (5,432 total).
   - Validation: First 150 sentences of official development split (923 total).
   - Test: **Complete official test split** (all 940 sentences, 24,497 tokens, 960 disease entities).
3. **MedMentions ST21pv:**
   - Preprocessing: Converted from official PubTator corpus partitioned by official PMID splits (2,635 train, 878 dev, 879 test PMIDs).
   - Training: First 400 sentences of official train partition (26,577 total).
   - Validation: First 150 sentences of official validation partition (8,770 total).
   - Test: **Project-created controlled subset** of 500 sentences from official test partition (8,863 total). *Not the full ST21pv test set.*

---

## 5. Limitations & Scientific Findings

1. **Corpus Density vs. Sample Budget:**
   MedMentions ST21pv contains over 200,000 mentions across 21 UMLS semantic types. When fine-tuned on a 400-sentence budget, the model achieves high precision on the surrogate disease class (66.67%), but recall remains low (14.77% on Disease, 5.18% on Chemical) due to high vocabulary sparsity.
2. **Homogeneous Disease Benchmarking:**
   NCBI Disease and BC5CDR both derive from PubMed literature and share MeSH-aligned disease concepts, resulting in consistent F1 scores (~75%–82%) under comparable sample budgets.

---

## 6. Non-Clinical Regulatory Notice

> [!CAUTION]
> **Research Prototype Only:** The models and findings in this report are strictly intended for scientific NLP research and clinical text analysis experimentation. None of these models have been cleared or approved by any medical regulatory authority (such as the FDA or EMA) as medical devices or clinical decision support systems.
