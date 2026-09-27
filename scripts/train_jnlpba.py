"""Train and evaluate PubMedBERT on JNLPBA / BioNLP 2004."""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.dataset import load_raw_dataset
from src.training.trainer import run_training_experiment


def main():
    parser = argparse.ArgumentParser(description="Train PubMedBERT on JNLPBA / BioNLP 2004")
    parser.add_argument("--model_path", type=str, default=str(root_dir / "pretrained_backbones" / "pubmedbert"))
    parser.add_argument("--data_dir", type=str, default=str(root_dir / "data" / "jnlpba" / "processed"))
    parser.add_argument("--exp_dir", type=str, default=str(root_dir / "experiments" / "jnlpba" / "pubmedbert"))
    parser.add_argument("--model_export_dir", type=str, default=str(root_dir / "models" / "jnlpba_pubmedbert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_samples", type=int, default=1500)
    parser.add_argument("--val_samples", type=int, default=300)
    parser.add_argument("--test_samples", type=int, default=None)  # None = full official test set (3,856 sent)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Training PubMedBERT on JNLPBA (BioNLP 2004)")
    print("=" * 60)
    print(f"Data directory:    {args.data_dir}")
    print(f"Model backbone:    {args.model_path}")
    print(f"Experiment output: {args.exp_dir}")
    print(f"Export target:     {args.model_export_dir}")
    print(f"Hyperparameters:   lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}, seed={args.seed}")
    print(f"Sample allocation: train={args.train_samples or 'FULL'}, val={args.val_samples or 'FULL'}, test={args.test_samples or 'FULL (3,856)'}")
    print("=" * 60 + "\n", flush=True)
    
    raw_splits, label2id, id2label = load_raw_dataset(args.data_dir)
    print(f"JNLPBA Splits: train={len(raw_splits['train'])}, valid={len(raw_splits['valid'])}, test={len(raw_splits['test'])}")
    print(f"Label map: {label2id}", flush=True)
    
    metrics = run_training_experiment(
        model_name_or_path=args.model_path,
        raw_splits=raw_splits,
        id2label=id2label,
        label2id=label2id,
        output_dir=args.exp_dir,
        experiment_name="jnlpba_pubmedbert",
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
            
    with open(export_path / "label.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
        
    with open(export_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    per_ent = metrics.get("per_entity_test_metrics", {})
    prot_f1 = per_ent.get("protein", {}).get("f1", 0.0)
    dna_f1 = per_ent.get("DNA", {}).get("f1", 0.0)
    rna_f1 = per_ent.get("RNA", {}).get("f1", 0.0)
    cell_type_f1 = per_ent.get("cell_type", {}).get("f1", 0.0)
    cell_line_f1 = per_ent.get("cell_line", {}).get("f1", 0.0)
    
    model_card = f"""# Model Card: JNLPBA / BioNLP 2004 PubMedBERT

- **Backbone:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **Domain:** Molecular Biology (MEDLINE abstracts)
- **Dataset:** JNLPBA (BioNLP 2004 Shared Task / GENIA)
- **Target Entities:** `protein`, `DNA`, `RNA`, `cell_type`, `cell_line`
- **Training Samples:** {args.train_samples}
- **Test Set Evaluated:** Full Official Test Set ({len(raw_splits['test']):,} sentences)
- **Test Precision:** {metrics['test_precision']:.4f}
- **Test Recall:** {metrics['test_recall']:.4f}
- **Test Entity F1:** {metrics['test_f1']:.4f}
- **Protein F1:** {prot_f1:.4f}
- **DNA F1:** {dna_f1:.4f}
- **RNA F1:** {rna_f1:.4f}
- **Cell Type F1:** {cell_type_f1:.4f}
- **Cell Line F1:** {cell_line_f1:.4f}
- **Training Time:** {metrics['training_time_minutes']:.2f} minutes
- **Parameters:** {metrics['parameters']:,}
"""
    with open(export_path / "model_card.md", "w", encoding="utf-8") as f:
        f.write(model_card)
        
    print(f"Best model exported to {export_path}")
    print(f"JNLPBA Experiment Complete. Test F1: {metrics['test_f1']:.4f}\n")


if __name__ == "__main__":
    main()
