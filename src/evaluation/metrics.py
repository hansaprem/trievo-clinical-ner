"""Evaluation metrics for Clinical NER using strict entity-level seqeval."""

from typing import Dict, List, Tuple
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.scheme import IOB2


def compute_metrics_fn_builder(id2label: Dict[int, str]):
    """Builds a compute_metrics function compatible with Hugging Face Trainer."""
    def compute_metrics(p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        predictions, labels = p
        preds = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [id2label[p_idx] for (p_idx, l_idx) in zip(pred_row, label_row) if l_idx != -100]
            for pred_row, label_row in zip(preds, labels)
        ]
        true_labels = [
            [id2label[l_idx] for (p_idx, l_idx) in zip(pred_row, label_row) if l_idx != -100]
            for pred_row, label_row in zip(preds, labels)
        ]
        
        # Calculate strict entity-level seqeval metrics
        prec = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        rec = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        f1 = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        
        return {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }
    return compute_metrics


def detailed_evaluation(preds_logits: np.ndarray, labels: np.ndarray, id2label: Dict[int, str]) -> Dict:
    """Computes detailed per-entity and overall strict entity-level metrics."""
    preds = np.argmax(preds_logits, axis=2)
    
    true_predictions = [
        [id2label[p_idx] for (p_idx, l_idx) in zip(pred_row, label_row) if l_idx != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l_idx] for (p_idx, l_idx) in zip(pred_row, label_row) if l_idx != -100]
        for pred_row, label_row in zip(preds, labels)
    ]
    
    prec = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
    rec = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
    f1 = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
    
    report_dict = classification_report(
        true_labels,
        true_predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        true_labels,
        true_predictions,
        mode="strict",
        scheme=IOB2,
        output_dict=False,
        zero_division=0
    )
    
    per_entity = {}
    for ent_type, vals in report_dict.items():
        if isinstance(vals, dict) and "f1-score" in vals and ent_type not in ["micro avg", "macro avg", "weighted avg"]:
            per_entity[ent_type] = {
                "precision": float(vals["precision"]),
                "recall": float(vals["recall"]),
                "f1": float(vals["f1-score"]),
                "support": int(vals["support"])
            }
    # Ensure default keys if not present
    for default_ent in ["Chemical", "Disease"]:
        if default_ent not in per_entity:
            per_entity[default_ent] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            
    return {
        "overall_precision": float(prec),
        "overall_recall": float(rec),
        "overall_f1": float(f1),
        "accuracy_note": "Accuracy was not used/reported because entity-level F1 is the primary NER evaluation metric.",
        "per_entity": per_entity,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "predictions_sample": true_predictions[:5],
        "labels_sample": true_labels[:5]
    }
