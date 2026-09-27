"""Training loop and experiment tracking for Clinical NER."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

from src.data.dataset import prepare_hf_dataset
from src.evaluation.metrics import compute_metrics_fn_builder, detailed_evaluation
from src.models.token_classifier import load_model_and_tokenizer


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def json_type_converter(obj: Any) -> Any:
    """Recursively converts numpy and int64/float64 types to native Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: json_type_converter(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_type_converter(i) for i in obj]
    return obj


def run_training_experiment(
    model_name_or_path: str,
    raw_splits: Dict[str, List[dict]],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    output_dir: str,
    experiment_name: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    max_length: int = 128,
    seed: int = 42,
    train_subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None
) -> Dict:
    """
    Executes a complete, reproducible clinical NER training and evaluation run.
    Saves metrics.json, config.json, training_history.json, classification_report.json,
    and best checkpoint.
    """
    set_seed(seed)
    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=======================================================", flush=True)
    print(f"Starting Experiment: {experiment_name}", flush=True)
    print(f"Model: {model_name_or_path}", flush=True)
    print(f"LR: {learning_rate}, Batch: {batch_size}, Epochs: {epochs}, Seed: {seed}", flush=True)
    print(f"=======================================================\n", flush=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, id2label, label2id)
    param_count = count_parameters(model)
    print(f"Trainable parameters: {param_count:,}", flush=True)
    
    # Prepare datasets
    train_records = raw_splits["train"][:train_subset_size] if train_subset_size else raw_splits["train"]
    valid_records = raw_splits["valid"][:val_subset_size] if val_subset_size else raw_splits["valid"]
    test_records = raw_splits["test"][:test_subset_size] if test_subset_size else raw_splits["test"]
    
    print(f"Dataset sizes -> Train: {len(train_records)}, Valid: {len(valid_records)}, Test: {len(test_records)}", flush=True)
    
    train_dataset = prepare_hf_dataset(train_records, tokenizer, label2id, max_length=max_length)
    valid_dataset = prepare_hf_dataset(valid_records, tokenizer, label2id, max_length=max_length)
    test_dataset = prepare_hf_dataset(test_records, tokenizer, label2id, max_length=max_length)
    
    # Dynamic padding collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    compute_metrics_fn = compute_metrics_fn_builder(id2label)
    
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=10,
        seed=seed,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=tokenizer
    )
    
    # Training
    t_start = time.time()
    train_result = trainer.train()
    training_time = time.time() - t_start
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes).", flush=True)
    
    # Detailed Validation Evaluation
    print("Evaluating on Validation Set...", flush=True)
    val_preds = trainer.predict(valid_dataset)
    val_detailed = detailed_evaluation(val_preds.predictions, val_preds.label_ids, id2label)
    
    # Detailed Test Evaluation (Held-out Test Set)
    print("Evaluating on Held-Out Test Set...", flush=True)
    t_inf_start = time.time()
    test_preds = trainer.predict(test_dataset)
    test_inf_time = time.time() - t_inf_start
    test_detailed = detailed_evaluation(test_preds.predictions, test_preds.label_ids, id2label)
    
    # Calculate inference latency per sentence
    avg_latency_ms = (test_inf_time / len(test_records)) * 1000 if len(test_records) > 0 else 0.0
    
    # Save the final best model weights and tokenizer
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    
    # Measure checkpoint size
    total_size_bytes = sum(f.stat().st_size for f in checkpoint_dir.glob("**/*") if f.is_file())
    checkpoint_size_mb = total_size_bytes / (1024 * 1024)
    
    # Save configurations
    config_dict = {
        "experiment_name": experiment_name,
        "model_name_or_path": model_name_or_path,
        "parameters": param_count,
        "tokenizer": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "max_sequence_length": max_length,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "random_seed": seed,
        "train_samples": len(train_records),
        "valid_samples": len(valid_records),
        "test_samples": len(test_records)
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
        
    # Save training history
    with open(exp_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
        
    # Save classification reports safely
    reports_payload = {
        "validation_report": json_type_converter(val_detailed["classification_report_dict"]),
        "validation_report_text": val_detailed["classification_report_text"],
        "test_report": json_type_converter(test_detailed["classification_report_dict"]),
        "test_report_text": test_detailed["classification_report_text"]
    }
    with open(exp_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(reports_payload, f, indent=2)
        
    # Summary Metrics
    metrics_dict = {
        "experiment_name": experiment_name,
        "model_name": model_name_or_path,
        "parameters": param_count,
        "training_time_seconds": round(training_time, 2),
        "training_time_minutes": round(training_time / 60, 2),
        "checkpoint_size_mb": round(checkpoint_size_mb, 2),
        "inference_latency_ms_per_sentence": round(avg_latency_ms, 2),
        "validation_precision": round(float(val_detailed["overall_precision"]), 4),
        "validation_recall": round(float(val_detailed["overall_recall"]), 4),
        "validation_f1": round(float(val_detailed["overall_f1"]), 4),
        "test_precision": round(float(test_detailed["overall_precision"]), 4),
        "test_recall": round(float(test_detailed["overall_recall"]), 4),
        "test_f1": round(float(test_detailed["overall_f1"]), 4),
        "per_entity_test_metrics": {
            ent: {
                "f1": round(float(vals["f1"]), 4),
                "precision": round(float(vals["precision"]), 4),
                "recall": round(float(vals["recall"]), 4),
                "support": int(vals["support"])
            }
            for ent, vals in test_detailed.get("per_entity", {}).items()
        },
        "accuracy_statement": "Accuracy was not reported because entity-level F1 is the primary NER evaluation metric."
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_type_converter(metrics_dict), f, indent=2)
        
    print(f"\n--- Completed {experiment_name} ---", flush=True)
    print(f"Validation F1: {val_detailed['overall_f1']:.4f}", flush=True)
    print(f"Test F1:       {test_detailed['overall_f1']:.4f}", flush=True)
    for ent, vals in test_detailed.get("per_entity", {}).items():
        if vals.get("support", 0) > 0:
            print(f"{ent} F1:    {vals['f1']:.4f}", flush=True)
    print(f"-----------------------------------------\n", flush=True)
    
    return metrics_dict
