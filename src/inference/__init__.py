"""Inference module for Clinical & Biomedical NER."""

from src.inference.adapter import ModelInferenceAdapter, RawEntitySpan
from src.inference.model_loader import LoadedModel, ModelRegistry
from src.inference.normalizer import normalize_label
from src.inference.pipeline import UnifiedClinicalNERPipeline
from src.inference.resolver import EntityResolver, UnifiedEntity
from src.inference.sliding_window import SlidingWindowTokenizer, TokenWindowChunk

__all__ = [
    "UnifiedClinicalNERPipeline",
    "ModelRegistry",
    "LoadedModel",
    "ModelInferenceAdapter",
    "RawEntitySpan",
    "EntityResolver",
    "UnifiedEntity",
    "normalize_label",
    "SlidingWindowTokenizer",
    "TokenWindowChunk",
]
