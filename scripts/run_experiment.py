"""Master Experiment Runner for TriEvo Clinical NER."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.dataset import load_raw_dataset
from src.training.trainer import run_training_experiment


def main():
    parser = argparse.ArgumentParser(description="Run Clinical NER Experiment")
    parser.add_argument("--model_id", type=str, required=True, help="HF model name or path")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name / directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_samples", type=int, default=None, help="Max train samples")
    parser.add_argument("--val_samples", type=int, default=None, help="Max val samples")
    parser.add_argument("--test_samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--data_dir", type=str, default=str(root_dir / "data" / "raw"))
    parser.add_argument("--output_base", type=str, default=str(root_dir / "experiments"))
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_dir}...")
    raw_splits, label2id, id2label = load_raw_dataset(args.data_dir)
    
    output_dir = Path(args.output_base) / args.exp_name
    
    metrics = run_training_experiment(
        model_name_or_path=args.model_id,
        raw_splits=raw_splits,
        id2label=id2label,
        label2id=label2id,
        output_dir=str(output_dir),
        experiment_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        train_subset_size=args.train_samples,
        val_subset_size=args.val_samples,
        test_subset_size=args.test_samples
    )
    
    print(f"Finished {args.exp_name} successfully.")


if __name__ == "__main__":
    main()
