# Qualitative and Quantitative Error Analysis Report

**Date:** 2026-09-24  
**Evaluated Model:** `./models/best_clinical_ner`  
**Test Subsample Size:** 200 sentences  
**Ground Truth Entities:** 271  
**Predicted Entities:** 281  

## 1. Error Distribution Overview

| Error Category | Count | Percentage of GT Entities |
| :--- | :--- | :--- |
| **Exact Span Matches (True Positives)** | 226 | 83.39% |
| **False Negatives (Missed Entities)** | 24 | 8.86% |
| **Boundary Mismatches (Partial Matches)** | 19 | 7.01% |
| **Type Confusions (Class Swaps)** | 2 | 0.74% |
| **False Positives (Spurious Detections)** | 35 | - |

## 2. Boundary Mismatch Patterns

Boundary errors typically arise in multi-word compound clinical descriptions where adjectival modifiers or anatomical locations are optionally included.

| Entity Type | Ground Truth Span | Model Predicted Span | Clinical Context |
| :--- | :--- | :--- | :--- |
| `Disease` | **ulcers** | *stress ulcers* | "Famotidine is a histamine H2 - receptor antagonist used in inpatient settings for preventi..." |
| `Disease` | **psychotic symptoms** | *psychotic* | "0 % of the patients were having current psychotic symptoms ...." |
| `Disease` | **Major depressive disorder** | *depressive disorder* | "Major depressive disorder ( OR = 2 ...." |
| `Disease` | **abnormal involuntary movements** | *involuntary* | "These results suggest that alterations in cerebellar sensory processing function , occurri..." |
| `Chemical` | **GR 82334** | *GR* | "injected with CYP with subsequently perfusion of bladder with P2X3 and NK1 receptors ' ant..." |
| `Chemical` | **Bisphenol A** | *A* | "Pubertal exposure to Bisphenol A increases anxiety - like behavior and decreases acetylcho..." |
| `Chemical` | **Bisphenol A** | *Bisphenol* | "The negative effects of Bisphenol A ( BPA ) on neurodevelopment and behaviors have been we..." |
| `Chemical` | **nitric oxide** | *nitric* | "The subcutaneous injection of isoproterenol ( 30 mg / kg ) into rats twice at an interval ..." |

## 3. False Negative Patterns (Missed Clinical Mentions)

Commonly missed entities frequently include specialized biochemical acronyms, rare syndromes, or ambiguous clinical descriptors.

| Missed Entity Type | Ground Truth Phrase | Sentence Context |
| :--- | :--- | :--- |
| `Chemical` | **sodium** | "Indomethacin induced hypotension in sodium and volume depleted rats ...." |
| `Chemical` | **sodium** | "After a single oral dose of 4 mg / kg indomethacin ( IDM ) to sodium and volume depleted r..." |
| `Chemical` | **IDM** | "After a single oral dose of 4 mg / kg indomethacin ( IDM ) to sodium and volume depleted r..." |
| `Chemical` | **angiotensin** | "Thus , indomethacin by inhibition of prostaglandin synthesis may diminish the blood pressu..." |
| `Chemical` | **prostaglandin** | "Thus , indomethacin by inhibition of prostaglandin synthesis may diminish the blood pressu..." |
| `Disease` | **edema** | "01 ). Histological changes evident in model and intervention groups rats ' bladder include..." |
| `Chemical` | **aspartate** | "The subcutaneous injection of isoproterenol ( 30 mg / kg ) into rats twice at an interval ..." |
| `Chemical` | **superoxide** | "The subcutaneous injection of isoproterenol ( 30 mg / kg ) into rats twice at an interval ..." |

## 4. False Positive Patterns (Spurious Predictions)

Spurious entities often correspond to general biological terms, anatomical organs, or non-drug chemical references.

| Predicted Type | Predicted Text | Sentence Context |
| :--- | :--- | :--- |
| `Disease` | *renal replacement* | "Scleroderma renal crisis ( SRC ) is a rare complication of systemic sclerosis ( SSc ) but ..." |
| `Disease` | *drug* | "METHODS : This was a cross - sectional study conducted concurrently at a teaching hospital..." |
| `Disease` | *major* | "Co - morbid major depressive disorder ( OR = 7 ...." |
| `Disease` | *Major* | "Major depressive disorder ( OR = 2 ...." |
| `Disease` | *M1* | "This was evident only when a sensory component was involved in the induction of plasticity..." |
| `Disease` | *PAS* | "To explore whether this benefit is linked to the restoration of sensorimotor plasticity of..." |
| `Disease` | *movements* | "These results suggest that alterations in cerebellar sensory processing function , occurri..." |
| `Disease` | *M1* | "These results suggest that alterations in cerebellar sensory processing function , occurri..." |

## 5. Type Confusion Analysis

Chemical vs. Disease confusion occurs predominantly when a compound is referenced as part of a pathological disease state (e.g. drug toxicity or chemical poisoning).

- Ground Truth: `Chemical` (**sodium**) -> Model: `Disease` (*sodium*)
- Ground Truth: `Chemical` (**methamphetamine**) -> Model: `Disease` (*methamphetamine*)

## 6. Synthesis & Mitigations

1. **Boundary Tuning:** Enhancing token boundary precision through CRF (Conditional Random Field) layers or subword pooling can resolve adjectival modifier boundary shifts.
2. **Domain-Specific Vocabulary:** Models with specialized biomedical subword vocabularies (e.g. PubMedBERT) significantly reduce out-of-vocabulary splits for complex pharmacology IUPAC names.
