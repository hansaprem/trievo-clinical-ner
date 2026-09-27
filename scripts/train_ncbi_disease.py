"""Train and evaluate PubMedBERT on NCBI Disease Corpus."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.dataset import load_raw_dataset
from src.training.trainer import run_training_experiment


def main():
    parser = argparse.ArgumentParser(description="Train PubMedBERT on NCBI Disease")
    parser.add_argument("--model_path", type=str, default=str(root_dir / "pretrained_backbones" / "pubmedbert"))
    parser.add_argument("--data_dir", type=str, default=str(root_dir / "data" / "ncbi_disease" / "processed"))
    parser.add_argument("--exp_dir", type=str, default=str(root_dir / "experiments" / "ncbi_disease" / "pubmedbert"))
    parser.add_argument("--model_export_dir", type=str, default=str(root_dir / "models" / "ncbi_disease_pubmedbert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_samples", type=int, default=400)
    parser.add_argument("--val_samples", type=int, default=150)
    parser.add_argument("--test_samples", type=int, default=None)  # None = full 940 test sentences
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("STEP 3: Training PubMedBERT on NCBI Disease Corpus")
    print("=" * 60)
    print(f"Data directory:    {args.data_dir}")
    print(f"Model backbone:    {args.model_path}")
    print(f"Experiment output: {args.exp_dir}")
    print(f"Export target:     {args.model_export_dir}")
    print(f"Hyperparameters:   lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, seed={args.seed}")
    print(f"Sample allocation: train={args.train_samples}, val={args.val_samples}, test={args.test_samples or 'FULL (940)'}")
    print("=" * 60 + "\n", flush=True)
    
    # Load dataset
    raw_splits, label2id, id2label = load_raw_dataset(args.data_dir)
    print(f"NCBI Disease Splits: train={len(raw_splits['train'])}, valid={len(raw_splits['valid'])}, test={len(raw_splits['test'])}")
    print(f"Label map: {label2id}", flush=True)
    
    # Run training experiment
    metrics = run_training_experiment(
        model_name_or_path=args.model_path,
        raw_splits=raw_splits,
        id2label=id2label,
        label2id=label2id,
        output_dir=args.exp_dir,
        experiment_name="ncbi_disease_pubmedbert",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        train_subset_size=args.train_samples,
        val_subset_size=args.val_samples,
        test_subset_size=args.test_samples
    )
    
    # Export best model separately
    export_path = Path(args.model_export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(args.exp_dir) / "checkpoint"
    print(f"\nExporting best model checkpoint from {checkpoint_dir} to {export_path}...", flush=True)
    for f in checkpoint_dir.glob("*"):
        if f.is_file():
            shutil.copy2(f, export_path / f.name)
            
    # Copy label map and metrics
    with open(export_path / "label.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
        
    with open(export_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # Create model card
    model_card = f"""# Model Card: NCBI Disease PubMedBERT

- **Backbone:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **Dataset:** NCBI Disease Corpus (793 PubMed abstracts)
- **Task:** Token-level Disease Named Entity Recognition (IOB2)
- **Labels:** `O`, `B-Disease`, `I-Disease`
- **Test Precision:** {metrics['test_precision']:.4f}
- **Test Recall:** {metrics['test_recall']:.4f}
- **Test Entity F1:** {metrics['test_f1']:.4f}
- **Training Time:** {metrics['training_time_minutes']:.2f} minutes
- **Parameters:** {metrics['parameters']:,}
"""
    with open(export_path / "model_card.md", "w", encoding="utf-8") as f:
        f.write(model_card)
        
    print(f"Best model exported to {export_path}")
    print(f"NCBI Disease Experiment Complete. Test F1: {metrics['test_f1']:.4f}\n")


if __name__ == "__main__":
    main()
