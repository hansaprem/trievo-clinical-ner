# Final Standardized 900-Sentence Clinical & Biomedical NER Benchmark Report

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
| **BC5CDR** | **900** | **0.7498** | **0.7876** | **0.7683** | Chemical, Disease |
| **NCBI Disease** | **900** | **0.7581** | **0.7581** | **0.7581** | Disease |
| **JNLPBA** | **900** | **0.6585** | **0.7569** | **0.7043** | Protein, DNA, RNA, Cell Type, Cell Line |
| **AnatEM** | **900** | **0.8144** | **0.7783** | **0.7959** | Anatomy |

**Macro-average F1 across datasets:** **0.7567**  
*(Macro-average Precision: 0.7452 | Macro-average Recall: 0.7702)*

> [!NOTE]
> **Evaluation Metric Policy:**  
> *Accuracy was not reported because entity-level F1 is the primary NER evaluation metric.* Background tokens (`O`) comprise ~90% of tokens in biomedical literature, which artificially inflates token-level accuracy while failing to measure boundary recognition. F1 is strictly calculated at the entity span level.

---

## 2. Per-Entity Metric Breakdown (Standardized 900 Test Sentences)

| Dataset | Entity Class | Precision | Recall | Entity F1 | Support (Gold Entities) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BC5CDR** | `Chemical` | 0.8160 | 0.8463 | **0.8309** | 833 |
| **BC5CDR** | `Disease` | 0.6681 | 0.7130 | **0.6898** | 655 |
| **NCBI Disease** | `Disease` | 0.7581 | 0.7581 | **0.7581** | 922 |
| **JNLPBA** | `DNA` | 0.6059 | 0.6842 | **0.6427** | 209 |
| **JNLPBA** | `RNA` | 0.7619 | 0.5000 | **0.6038** | 32 |
| **JNLPBA** | `cell_line` | 0.4575 | 0.6863 | **0.5490** | 102 |
| **JNLPBA** | `cell_type` | 0.7347 | 0.7382 | **0.7365** | 424 |
| **JNLPBA** | `protein` | 0.6645 | 0.7897 | **0.7217** | 1,179 |
| **AnatEM** | `Anatomy` | 0.8144 | 0.7783 | **0.7959** | 1,105 |

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
| **BC5CDR** | 900 | 17,574 | 1,488 | 1,563 | [`models/bc5cdr_pubmedbert`](file:///C:/Users/khan computer/.gemini/antigravity/scratch/trievo-clinical-ner/models/bc5cdr_pubmedbert) |
| **NCBI Disease** | 900 | 23,561 | 922 | 922 | [`models/ncbi_disease_pubmedbert`](file:///C:/Users/khan computer/.gemini/antigravity/scratch/trievo-clinical-ner/models/ncbi_disease_pubmedbert) |
| **JNLPBA** | 900 | 23,074 | 1,946 | 2,237 | [`models/jnlpba_pubmedbert`](file:///C:/Users/khan computer/.gemini/antigravity/scratch/trievo-clinical-ner/models/jnlpba_pubmedbert) |
| **AnatEM** | 900 | 24,526 | 1,105 | 1,056 | [`models/anatem_pubmedbert`](file:///C:/Users/khan computer/.gemini/antigravity/scratch/trievo-clinical-ner/models/anatem_pubmedbert) |
| **Total Benchmark** | **3,600** | **88,735** | **5,461** | **5,778** | **4 Distinct Dedicated Model Heads** |

---

## 5. Non-Clinical Regulatory Notice

> [!CAUTION]
> **Research Prototype Only:** The models, checkpoints, and evaluations documented herein were developed solely for academic NLP research and algorithmic comparison. None of these models are approved or certified as medical devices, diagnostic aids, or clinical decision support software by the FDA, EMA, or any national health regulatory authority.
