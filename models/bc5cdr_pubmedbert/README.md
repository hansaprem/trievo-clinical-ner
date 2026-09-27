# Best Clinical NER Checkpoint

This directory contains the production weights, tokenizer, configuration, and label mapping for the best-performing clinical NER model: **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`).

## Quick Usage via Python
```python
from src.inference.pipeline import ClinicalNERPipeline

pipeline = ClinicalNERPipeline("models/best_clinical_ner")
res = pipeline.extract_entities("Patient has acute myocardial infarction and takes aspirin.")
print(res)
```

For complete benchmarking data and specifications, see `model_card.md`.
