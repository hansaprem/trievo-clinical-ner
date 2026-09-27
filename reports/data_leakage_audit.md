# Data Leakage & Test Partition Isolation Audit Report

**Date:** 2026-09-27  
**Standardized Benchmark Protocol:** Exactly 900 held-out test sentences per dataset  
**Sampling Determinism:** Random Seed = 42  
**Audit Standard:** Strict Exact Text-Hash Isolation, 0 Duplicates, 0 Invalid BIO Transitions  

---

## 1. Audit Summary Matrix

| Dataset | Original Test Pool | Standardized Test Size | Train/Test Overlap | Valid/Test Overlap | Internal Test Duplicates | Invalid BIO Transitions | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC5CDR** | 5,865 | **900** | **0** | **0** | **0** | **0** | **PASSED** |
| **NCBI Disease** | 940 | **900** | **0** | **0** | **0** | **0** | **PASSED** |
| **JNLPBA** | 3,856 | **900** | **0** | **0** | **0** | **0** | **PASSED** |
| **AnatEM** | 3,830 | **900** | **0** | **0** | **0** | **0** | **PASSED** |

---

## 2. Methodology & Guarantees

1. **Source Isolation:** Every selected sentence was sampled strictly from its official competition test partition.
2. **Deterministic Sampling:** Seed 42 was applied to a deterministic non-overlapping candidate list. Indices are frozen in `reports/final_900_sentence_manifest.json`.
3. **Zero Contamination:** No training or validation sample from any split was admitted into any test partition.
4. **Zero Entity Distortion:** All character offsets and entity boundaries are 100% identical to official annotations.
5. **No Synthetic Data:** All sentences are authentic, human-curated biomedical texts.
