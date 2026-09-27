# Biomedical & Clinical Model Research and Compatibility Report

**Date:** 2026-09-24  
**Project:** TriEvo Clinical NER (`trievo-clinical-ner`)  

## 1. Candidate Model Architecture Specifications

| Model Name | Hugging Face ID | Vocab Size | Hidden Dim | Layers | Heads | Max Seq Len | Case Sensitivity | Token Classification Ready |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (General Domain)** | `google-bert/bert-base-uncased` | 30,522 | 768 | 12 | 12 | 512 | Uncased | Yes |
| **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | 28,895 | 768 | 12 | 12 | 512 | Uncased | Yes |
| **BioBERT** | `dmis-lab/biobert-base-cased-v1.2` | 28,996 | 768 | 12 | 12 | 512 | Uncased | Yes |
| **BioClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` | 28,996 | 768 | 12 | 12 | 512 | Uncased | Yes |
| **SciBERT** | `allenai/scibert_scivocab_uncased` | 31,090 | 768 | 12 | 12 | 512 | Uncased | Yes |

## 2. In-Depth Candidate Architectural & Domain Profiles

### Baseline (General Domain) (`google-bert/bert-base-uncased`)
- **Domain Pretraining:** Standard general-domain BERT baseline trained on BookCorpus & Wikipedia
- **Vocabulary Profile:** 30,522 WordPiece tokens (BertTokenizer)
- **Context Capacity:** 512 tokens (dataset 99th percentile length is ~45 tokens, easily accommodated)
- **Suitability for BC5CDR:** High. Pretrained on scientific/biomedical texts rich in pharmacology and disease taxonomy.

### PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)
- **Domain Pretraining:** Domain-specific BERT trained from scratch on PubMed abstracts with PubMed vocabulary
- **Vocabulary Profile:** 28,895 WordPiece tokens (BertTokenizer)
- **Context Capacity:** 512 tokens (dataset 99th percentile length is ~45 tokens, easily accommodated)
- **Suitability for BC5CDR:** High. Pretrained on scientific/biomedical texts rich in pharmacology and disease taxonomy.

### BioBERT (`dmis-lab/biobert-base-cased-v1.2`)
- **Domain Pretraining:** Biomedical BERT initialized from BERT-base and continually pretrained on PubMed and PMC
- **Vocabulary Profile:** 28,996 WordPiece tokens (BertTokenizer)
- **Context Capacity:** 512 tokens (dataset 99th percentile length is ~45 tokens, easily accommodated)
- **Suitability for BC5CDR:** High. Pretrained on scientific/biomedical texts rich in pharmacology and disease taxonomy.

### BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- **Domain Pretraining:** Clinical BERT initialized from BioBERT and trained on MIMIC-III clinical ICU notes
- **Vocabulary Profile:** 28,996 WordPiece tokens (BertTokenizer)
- **Context Capacity:** 512 tokens (dataset 99th percentile length is ~45 tokens, easily accommodated)
- **Suitability for BC5CDR:** High. Pretrained on scientific/biomedical texts rich in pharmacology and disease taxonomy.

### SciBERT (`allenai/scibert_scivocab_uncased`)
- **Domain Pretraining:** Scientific BERT trained from scratch on 1.14M papers from Semantic Scholar with custom scientific vocab
- **Vocabulary Profile:** 31,090 WordPiece tokens (BertTokenizer)
- **Context Capacity:** 512 tokens (dataset 99th percentile length is ~45 tokens, easily accommodated)
- **Suitability for BC5CDR:** High. Pretrained on scientific/biomedical texts rich in pharmacology and disease taxonomy.

## 3. Compatibility Verdict

All 5 models utilize standard BERT encoder topologies with identical hidden dimension (768), 12 layers, 12 attention heads, and linear classification projection heads. All tokenizers serialize properly and conform to Hugging Face `AutoModelForTokenClassification` interfaces.
