"""Model Loader and Registry for Clinical NER Checkpoints.

Manages the lifecycle, loading, and device placement of the four trained
PubMedBERT token classification models:
  - BC5CDR (Chemical, Disease)
  - NCBI Disease (Disease)
  - JNLPBA (protein, cell_type, DNA, cell_line, RNA)
  - AnatEM (Anatomy)
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("trievo_clinical_ner_loader")

DEFAULT_HF_NAMESPACE = "hansaprem703"


@dataclass
class LoadedModel:
    """Container for a loaded token classification model and tokenizer."""
    name: str
    model_dir: Path
    entity_types: List[str]
    model: AutoModelForTokenClassification
    tokenizer: AutoTokenizer
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    device: torch.device


MODEL_METADATA = {
    "BC5CDR": {
        "dir_name": "bc5cdr_pubmedbert",
        "hf_repo_name": "trievo-bc5cdr-pubmedbert",
        "entity_types": ["Chemical", "Disease"],
    },
    "NCBI Disease": {
        "dir_name": "ncbi_disease_pubmedbert",
        "hf_repo_name": "trievo-ncbi-disease-pubmedbert",
        "entity_types": ["Disease"],
    },
    "JNLPBA": {
        "dir_name": "jnlpba_pubmedbert",
        "hf_repo_name": "trievo-jnlpba-pubmedbert",
        "entity_types": ["protein", "cell_type", "DNA", "cell_line", "RNA"],
    },
    "AnatEM": {
        "dir_name": "anatem_pubmedbert",
        "hf_repo_name": "trievo-anatem-pubmedbert",
        "entity_types": ["Anatomy"],
    },
}


class ModelRegistry:
    """Registry that caches and serves initialized models and tokenizers."""

    def __init__(
        self,
        models_root: Optional[Path] = None,
        device: Optional[str] = None,
        preload: bool = False,
    ):
        self.models_root = Path(models_root) if models_root else (PROJECT_ROOT / "models")
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._cache: Dict[str, LoadedModel] = {}

        if preload:
            self.load_all()

    def get_supported_models(self) -> List[str]:
        """Returns the list of supported model identifier strings."""
        return list(MODEL_METADATA.keys())

    def is_loaded(self, name: str) -> bool:
        """Returns True if the specified model is currently loaded in memory."""
        return name in self._cache

    def get(self, name: str) -> LoadedModel:
        """Retrieves a loaded model from cache, loading it from disk if necessary."""
        if name not in MODEL_METADATA:
            raise KeyError(
                f"Unknown model '{name}'. Supported models are: {list(MODEL_METADATA.keys())}"
            )

        if name not in self._cache:
            self._cache[name] = self._load_model(name)

        return self._cache[name]

    def load_all(self) -> Dict[str, LoadedModel]:
        """Eagerly loads all supported models into memory."""
        for name in MODEL_METADATA:
            if name not in self._cache:
                self._cache[name] = self._load_model(name)
        return self._cache

    def unload(self, name: str) -> None:
        """Unloads a model from memory and clears GPU/CPU cache."""
        if name in self._cache:
            del self._cache[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Unloads all models from memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, name: str) -> LoadedModel:
        meta = MODEL_METADATA[name]
        model_dir = self.models_root / meta["dir_name"]
        weights_file = model_dir / "model.safetensors"
        label_file = model_dir / "label.json"

        # Check if local model directory and weights exist; if missing, download from Hugging Face
        if not (model_dir.exists() and weights_file.exists() and label_file.exists()):
            hf_namespace = os.getenv("HF_MODEL_NAMESPACE", DEFAULT_HF_NAMESPACE)
            repo_id = f"{hf_namespace}/{meta['hf_repo_name']}"
            logger.info(
                "Local model '%s' missing at %s. Downloading from Hugging Face (%s)...",
                name,
                model_dir,
                repo_id,
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                token=os.getenv("HF_TOKEN") or None,
                ignore_patterns=[
                    "checkpoint-*",
                    "optimizer.pt",
                    "*.bin",
                    "trainer_state.json",
                    "training_args.bin",
                    "training_config.json",
                ],
            )
            logger.info("Successfully cached '%s' from Hugging Face to %s", name, model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found for '{name}' at expected path: {model_dir}"
            )

        # 1. Load label mapping
        if not label_file.exists():
            raise FileNotFoundError(f"Missing label.json for '{name}' in {model_dir}")
        with open(label_file, "r", encoding="utf-8") as f:
            raw_label2id = json.load(f)
        label2id = {k: int(v) for k, v in raw_label2id.items()}
        id2label = {int(v): k for k, v in raw_label2id.items()}

        # 2. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # 3. Load model
        config = AutoConfig.from_pretrained(str(model_dir))
        config.id2label = id2label
        config.label2id = label2id

        model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir),
            config=config,
        )
        model.to(self.device)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        return LoadedModel(
            name=name,
            model_dir=model_dir,
            entity_types=meta["entity_types"],
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            label2id=label2id,
            device=self.device,
        )
