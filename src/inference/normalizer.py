"""Entity Label Normalizer for Clinical NER Pipeline.

Maps dataset-specific annotation tags into a consistent, unified uppercase
biomedical ontology schema.
"""

from typing import Dict

STANDARD_LABEL_MAPPING: Dict[str, str] = {
    # Pharmacology
    "chemical": "CHEMICAL",
    "drug": "CHEMICAL",
    # Pathology
    "disease": "DISEASE",
    "disorder": "DISEASE",
    # Anatomy & Morphology
    "anatomy": "ANATOMY",
    "anatomical": "ANATOMY",
    # Molecular Biology (JNLPBA)
    "protein": "PROTEIN",
    "cell_type": "CELL_TYPE",
    "dna": "DNA",
    "cell_line": "CELL_LINE",
    "rna": "RNA",
}

SUPPORTED_NORMALIZED_LABELS = [
    "CHEMICAL",
    "DISEASE",
    "ANATOMY",
    "PROTEIN",
    "CELL_TYPE",
    "DNA",
    "CELL_LINE",
    "RNA",
]


def normalize_label(raw_label: str) -> str:
    """
    Normalizes a dataset-specific label into the unified clinical NER schema.

    Args:
        raw_label: Raw string label from model checkpoint (e.g. 'Chemical', 'protein').

    Returns:
        Standardized uppercase label string (e.g. 'CHEMICAL', 'PROTEIN').
    """
    clean = raw_label.strip()
    if clean.startswith("B-") or clean.startswith("I-"):
        clean = clean[2:]

    lower_key = clean.lower()
    return STANDARD_LABEL_MAPPING.get(lower_key, clean.upper())
