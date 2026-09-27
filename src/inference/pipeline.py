"""Unified Multi-Model Clinical NER Inference Pipeline.

Orchestrates the four trained PubMedBERT token classification models:
  - BC5CDR (Chemical, Disease)
  - NCBI Disease (Disease)
  - JNLPBA (Protein, DNA, RNA, Cell Type, Cell Line)
  - AnatEM (Anatomy)

Provides end-to-end multi-entity extraction with character offset alignment,
sliding-window support for arbitrary text lengths, and deterministic conflict resolution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch

from src.inference.adapter import ModelInferenceAdapter, RawEntitySpan
from src.inference.model_loader import LoadedModel, ModelRegistry
from src.inference.resolver import EntityResolver, UnifiedEntity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class UnifiedClinicalNERPipeline:
    """Production modular Clinical NER orchestrator over all trained checkpoints."""

    def __init__(
        self,
        models_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        enabled_models: Optional[List[str]] = None,
        window_size: int = 384,
        stride: int = 64,
        preload: bool = True,
    ):
        """
        Initializes the unified multi-model clinical NER pipeline.

        Args:
            models_root: Directory containing model checkpoints (default: project / 'models').
            device: 'cuda' or 'cpu'. Auto-detected if None.
            enabled_models: List of model names to use. Defaults to all 4 models.
            window_size: Maximum token window size (default: 384, max: 512).
            stride: Overlap stride between windows (default: 64).
            preload: If True, eagerly loads all enabled models into memory.
        """
        self.models_root = Path(models_root) if models_root else (PROJECT_ROOT / "models")
        self.registry = ModelRegistry(
            models_root=self.models_root,
            device=device,
            preload=False,
        )
        self.device = self.registry.device
        self.window_size = window_size
        self.stride = stride

        if enabled_models is None:
            self.enabled_models = self.registry.get_supported_models()
        else:
            supported = set(self.registry.get_supported_models())
            for m in enabled_models:
                if m not in supported:
                    raise ValueError(f"Model '{m}' is not supported. Supported: {list(supported)}")
            self.enabled_models = list(enabled_models)

        self.resolver = EntityResolver()
        self.adapters: Dict[str, ModelInferenceAdapter] = {}

        if preload:
            self._init_adapters()

    def _init_adapters(self) -> None:
        """Initializes model inference adapters for all enabled models."""
        for name in self.enabled_models:
            if name not in self.adapters:
                loaded = self.registry.get(name)
                self.adapters[name] = ModelInferenceAdapter(
                    loaded_model=loaded,
                    window_size=self.window_size,
                    stride=self.stride,
                )

    def extract_entities(
        self,
        text: str,
        include_raw: bool = False,
        extended_provenance: bool = False,
    ) -> Dict:
        """
        Extracts multi-type biomedical entities from input text across all models.

        Args:
            text: Raw clinical or biomedical text.
            include_raw: If True, attaches raw unmerged predictions for debugging.
            extended_provenance: If True, includes sub-spans and per-model confidences.

        Returns:
            Dictionary strictly adhering to the schema:
            {
                "text": "...",
                "entities": [
                    {
                        "text": "Aspirin",
                        "label": "CHEMICAL",
                        "start": 0,
                        "end": 7,
                        "source_model": "BC5CDR",
                        "confidence": 0.9852
                    }
                ]
            }
        """
        if not text or not text.strip():
            result = {
                "text": text,
                "entities": [],
            }
            if include_raw:
                result["raw_entities_by_model"] = {m: [] for m in self.enabled_models}
            return result

        self._init_adapters()

        all_raw_spans: List[RawEntitySpan] = []
        raw_by_model: Dict[str, List[Dict]] = {}

        # 1. Run each model independently
        for name in self.enabled_models:
            adapter = self.adapters[name]
            raw_spans = adapter.extract_entities(text)
            all_raw_spans.extend(raw_spans)
            if include_raw:
                raw_by_model[name] = [
                    {
                        "text": s.text,
                        "raw_label": s.raw_label,
                        "start": s.start,
                        "end": s.end,
                        "confidence": s.confidence,
                    }
                    for s in raw_spans
                ]

        # 2. Resolve conflicts, overlaps, and deduplicate
        resolved: List[UnifiedEntity] = self.resolver.resolve(all_raw_spans, text)

        # 3. Format according to schema
        if extended_provenance:
            entity_dicts = [ent.to_extended_dict() for ent in resolved]
        else:
            entity_dicts = [ent.to_dict() for ent in resolved]

        result = {
            "text": text,
            "entities": entity_dicts,
        }

        if include_raw:
            result["raw_entities_by_model"] = raw_by_model
            result["metadata"] = {
                "device": str(self.device),
                "models_used": self.enabled_models,
            }

        return result

    def predict(self, text: str, **kwargs) -> Dict:
        """Alias for extract_entities."""
        return self.extract_entities(text, **kwargs)
