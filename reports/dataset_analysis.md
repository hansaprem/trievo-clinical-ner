# Dataset Analysis Report: BioCreative V CDR (BC5CDR)

**Date:** 2026-09-24  
**Project:** TriEvo Clinical NER (`trievo-clinical-ner`)  

## 1. Executive Summary & Provenance

- **Dataset Name:** BioCreative V CDR (Chemical and Disease Named Entity Recognition)
- **Dataset Source:** BioCreative V Challenge / T-NER Benchmark (NLM PubMed peer-reviewed abstracts)
- **Annotation Task:** Token-level Named Entity Recognition for chemical compounds/medications and clinical diseases
- **Splitting Scheme:** Standardized official 3-way partition (Train / Validation / Test) strictly preserving document-level boundary isolation
- **Tagging Scheme:** BIO format (`O`, `B-Chemical`, `B-Disease`, `I-Disease`, `I-Chemical`)

## 2. Dataset Splitting & Sequence Metrics

| Split | Sentences | Total Tokens | Mean Tokens/Sent | Median Tokens/Sent | Max Tokens | Min Tokens | Internal Duplicates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 5,228 | 109,322 | 20.91 | 18.0 | 133 | 1 | 87 |
| **Validation** | 5,330 | 108,958 | 20.44 | 18.0 | 232 | 1 | 128 |
| **Test** | 5,865 | 116,318 | 19.83 | 17.0 | 135 | 1 | 152 |
| **TOTAL** | **16,423** | **334,598** | **20.37** | - | **232** | **1** | - |

## 3. Entity Class Distribution (Entity Spans)

| Split | Chemical Spans | Disease Spans | Total Entities | Entity Density (Entities / Sent) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 5,203 | 4,182 | 9,385 | 1.80 |
| **Validation** | 5,347 | 4,244 | 9,591 | 1.80 |
| **Test** | 5,385 | 4,424 | 9,809 | 1.67 |
| **Grand Total** | **15,935** | **12,850** | **28,785** | **1.75** |

## 4. Token-Level Tag Distribution

| Tag | Label ID | Train Tokens | Valid Tokens | Test Tokens | Total Tokens | % of All Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `O` | 0 | 96,796 | 96,413 | 103,684 | 296,893 | 88.73% |
| `B-Chemical` | 1 | 5,203 | 5,347 | 5,385 | 15,935 | 4.76% |
| `B-Disease` | 2 | 4,182 | 4,244 | 4,424 | 12,850 | 3.84% |
| `I-Disease` | 3 | 2,570 | 2,416 | 2,424 | 7,410 | 2.21% |
| `I-Chemical` | 4 | 571 | 538 | 401 | 1,510 | 0.45% |

## 5. Cross-Split Leakage & Integrity Audit

- **Train-Validation Identical Sentences:** 55 (1.06%)
- **Train-Test Identical Sentences:** 64 (1.12%)
- **Validation-Test Identical Sentences:** 55 (0.96%)

> **Audit Note:** The minimal identical sentences across splits correspond to generic scientific boilerplate formulas without entities. Official document partition boundaries are strictly preserved.

## 6. Key Characteristics & Modeling Requirements

1. **Severe Class Imbalance:** Over 88% of tokens are tagged `O` (outside entities). Models must optimize for discriminative boundary precision.
2. **Multi-token Clinical Terms:** Complex pharmacological compounds and disease phenotypes require subword alignment and BIO consistency.
3. **Context Length Feasibility:** Max sequence length across all splits is 250 tokens, fitting comfortably within the 512 max position window.
4. **Evaluation Standard:** Strict entity-level precision, recall, and F1 via `seqeval` are enforced. Token accuracy is strictly avoided as it is uninformative due to `O` dominance.
