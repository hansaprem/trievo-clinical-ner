"""Command-line Interface for Unified Multi-Model Clinical NER.

Usage examples:
  python scripts/cli_inference.py --text "Aspirin was administered to a patient with acute pneumonia."
  python scripts/cli_inference.py --file input.txt --output result.json
  python scripts/cli_inference.py --text "tacrolimus therapy" --models bc5cdr anatem
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import UnifiedClinicalNERPipeline


MODEL_ALIAS_MAP = {
    "bc5cdr": "BC5CDR",
    "ncbi": "NCBI Disease",
    "ncbi_disease": "NCBI Disease",
    "ncbi-disease": "NCBI Disease",
    "jnlpba": "JNLPBA",
    "anatem": "AnatEM",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Multi-Model Clinical NER Inference Engine"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        "-t",
        type=str,
        help="Raw text string for biomedical entity extraction",
    )
    group.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to text file to process",
    )

    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=None,
        help="Subset of models to run (e.g. bc5cdr ncbi_disease jnlpba anatem). Defaults to all.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device (default: auto-detect CUDA if available)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save JSON output",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Include unmerged raw model predictions in output for debugging",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include extended provenance details (sub-spans, per-model confidences)",
    )
    return parser.parse_args()


def resolve_model_names(requested: Optional[List[str]]) -> Optional[List[str]]:
    if not requested:
        return None
    resolved = []
    for r in requested:
        low = r.lower().strip()
        if low in MODEL_ALIAS_MAP:
            resolved.append(MODEL_ALIAS_MAP[low])
        else:
            resolved.append(r)
    return resolved


def main():
    args = parse_args()

    # 1. Read input text
    if args.text is not None:
        input_text = args.text
    else:
        if not args.file.exists():
            print(f"Error: File '{args.file}' does not exist.", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            input_text = f.read()

    # 2. Resolve models
    enabled_models = resolve_model_names(args.models)

    # 3. Initialize pipeline
    print(f"Initializing Unified Clinical NER Pipeline...", file=sys.stderr)
    pipeline = UnifiedClinicalNERPipeline(
        device=args.device,
        enabled_models=enabled_models,
        preload=True,
    )

    # 4. Run inference
    print(f"Extracting entities (input length: {len(input_text)} characters)...", file=sys.stderr)
    result = pipeline.extract_entities(
        input_text,
        include_raw=args.raw,
        extended_provenance=args.extended,
    )

    # 5. Output results
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(json_str)

    # Print summary table to stderr for human readability
    entities = result.get("entities", [])
    print(f"\n--- Extracted {len(entities)} Entities ---", file=sys.stderr)
    if entities:
        print(f"{'Text':<25} | {'Label':<12} | {'Span':<10} | {'Confidence':<10} | {'Source Model'}", file=sys.stderr)
        print("-" * 75, file=sys.stderr)
        for e in entities:
            span_str = f"[{e['start']}:{e['end']}]"
            print(f"{e['text'][:25]:<25} | {e['label']:<12} | {span_str:<10} | {e['confidence']:<10.4f} | {e['source_model']}", file=sys.stderr)
    print("", file=sys.stderr)


if __name__ == "__main__":
    main()
