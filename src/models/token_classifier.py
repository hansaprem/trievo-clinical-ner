"""Model loading and token classification head configuration."""

from typing import Dict, Tuple
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    classifier_dropout: float = 0.1
) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """Loads pretrained BERT backbone and adds linear token classification head."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        classifier_dropout=classifier_dropout
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config
    )
    
    return model, tokenizer
