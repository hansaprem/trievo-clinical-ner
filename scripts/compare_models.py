"""Comparative analysis and reporting across all trained NER models."""

import json
import os
from pathlib import Path
import pandas as pd


def generate_comparison(experiments_dir: str, output_dir: str):
    exp_path = Path(experiments_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    experiment_dirs = [d for d in exp_path.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    
    if not experiment_dirs:
        print("No completed experiments found with metrics.json.")
        return None
        
    records = []
    baseline_f1 = None
    
    # First find baseline F1 if baseline exists
    for ed in experiment_dirs:
        with open(ed / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        if ed.name == "baseline" or "baseline" in m.get("experiment_name", "").lower():
            baseline_f1 = m["test_f1"]
            break
            
    for ed in experiment_dirs:
        with open(ed / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        with open(ed / "config.json", "r", encoding="utf-8") as f:
            c = json.load(f)
            
        test_f1 = m["test_f1"]
        abs_diff = round((test_f1 - baseline_f1) * 100, 2) if baseline_f1 is not None else 0.0
        rel_diff = round(((test_f1 - baseline_f1) / baseline_f1) * 100, 2) if baseline_f1 and baseline_f1 > 0 else 0.0
        
        row = {
            "Experiment": m["experiment_name"],
            "Model Backbone": m["model_name"],
            "Parameters": c.get("parameters", m.get("parameters", 0)),
            "Train Samples": c.get("train_samples", "Full"),
            "Epochs": c.get("epochs", 3),
            "Learning Rate": c.get("learning_rate", 3e-5),
            "Train Time (min)": m["training_time_minutes"],
            "Inference Latency (ms/sent)": m["inference_latency_ms_per_sentence"],
            "Checkpoint (MB)": m["checkpoint_size_mb"],
            "Val Precision": m["validation_precision"],
            "Val Recall": m["validation_recall"],
            "Val F1": m["validation_f1"],
            "Test Precision": m["test_precision"],
            "Test Recall": m["test_recall"],
            "Test F1": m["test_f1"],
            "Abs F1 vs Base (pts)": abs_diff,
            "Rel F1 vs Base (%)": rel_diff,
            "Chemical F1": m["per_entity_test_metrics"]["Chemical_f1"],
            "Chemical Prec": m["per_entity_test_metrics"]["Chemical_precision"],
            "Chemical Rec": m["per_entity_test_metrics"]["Chemical_recall"],
            "Disease F1": m["per_entity_test_metrics"]["Disease_f1"],
            "Disease Prec": m["per_entity_test_metrics"]["Disease_precision"],
            "Disease Rec": m["per_entity_test_metrics"]["Disease_recall"]
        }
        records.append(row)
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="Test F1", ascending=False)
    
    # Save CSV and JSON
    csv_file = out_path / "model_comparison.csv"
    json_file = out_path / "model_comparison.json"
    
    df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
        
    print(f"Generated comparison at:\n  - {csv_file}\n  - {json_file}")
    print("\n--- MODEL COMPARISON SUMMARY ---")
    print(df[["Experiment", "Model Backbone", "Val F1", "Test F1", "Chemical F1", "Disease F1", "Train Time (min)"]].to_string(index=False))
    return df


if __name__ == "__main__":
    experiments_dir = r"C:\Users\khan computer\.gemini\antigravity\scratch\trievo-clinical-ner\experiments"
    generate_comparison(experiments_dir, experiments_dir)
