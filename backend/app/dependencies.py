"""Application Dependencies, Configuration, and Pipeline Lifecycle.

Manages environment settings and dependency injection for the FastAPI application.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import HTTPException, Request, status
import torch

from src.inference.pipeline import UnifiedClinicalNERPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings:
    """Application settings read from environment variables with sensible defaults."""

    def __init__(self):
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0" if (os.getenv("PORT") or os.getenv("RENDER")) else "127.0.0.1")
        self.api_port: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
        self.device_setting: str = os.getenv("DEVICE", "auto").lower()

        # Resolve compute device
        if self.device_setting == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device_setting == "cpu":
            self.device = "cpu"
        else:  # auto
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model directory
        models_dir_env = os.getenv("MODELS_DIR")
        self.models_dir: Path = Path(models_dir_env) if models_dir_env else (PROJECT_ROOT / "models")

        # CORS origins (combines FRONTEND_ORIGIN, CORS_ORIGINS, and local dev ports)
        origins: List[str] = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

        frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
        if frontend_origin:
            cleaned = frontend_origin.rstrip("/")
            if cleaned not in origins:
                origins.append(cleaned)

        raw_cors = os.getenv("CORS_ORIGINS", "").strip()
        if raw_cors:
            for orig in raw_cors.split(","):
                clean_orig = orig.strip().rstrip("/")
                if clean_orig and clean_orig not in origins:
                    origins.append(clean_orig)

        self.cors_origins: List[str] = origins


settings = Settings()


def get_settings() -> Settings:
    """Dependency provider for application settings."""
    return settings


def get_pipeline(request: Request) -> UnifiedClinicalNERPipeline:
    """
    Dependency provider that yields the single pre-warmed UnifiedClinicalNERPipeline
    from the application state. Avoids reloading models per request.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clinical NER pipeline is not initialized or still warming up.",
        )
    return pipeline
