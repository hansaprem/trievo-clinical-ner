"""Pydantic Request and Response Schemas for Clinical NER API.

Strictly validates input payloads and ensures output entities conform to
character offset constraints and clinical ontology standards.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictRequest(BaseModel):
    """Input payload for clinical entity extraction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The patient was prescribed aspirin for pneumonia."
            }
        }
    )

    text: str = Field(
        ...,
        description="Raw clinical or biomedical text to extract entities from.",
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if v is None:
            raise ValueError("Input text cannot be null.")
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Input text cannot be empty or whitespace-only.")
        return v


class Entity(BaseModel):
    """Extracted biomedical entity with exact character offsets and model provenance."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "aspirin",
                "label": "CHEMICAL",
                "start": 27,
                "end": 34,
                "source_model": "BC5CDR",
                "confidence": 0.9455,
            }
        }
    )

    text: str = Field(..., description="Verbatim entity substring extracted from text.")
    label: str = Field(
        ...,
        description="Standardized uppercase clinical entity type (CHEMICAL, DISEASE, ANATOMY, PROTEIN, etc.).",
    )
    start: int = Field(..., ge=0, description="0-indexed start character offset in original text.")
    end: int = Field(..., gt=0, description="0-indexed exclusive end character offset in original text.")
    source_model: str = Field(
        ...,
        description="Name of the model(s) that predicted or contributed to this entity.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Authentic model softmax probability confidence score (0.0 to 1.0).",
    )

    @model_validator(mode="after")
    def validate_span_indices(self) -> "Entity":
        if self.end <= self.start:
            raise ValueError(f"Entity end offset ({self.end}) must be strictly greater than start offset ({self.start}).")
        return self


class PredictResponse(BaseModel):
    """Output schema for clinical entity extraction."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The patient was prescribed aspirin for pneumonia.",
                "entities": [
                    {
                        "text": "aspirin",
                        "label": "CHEMICAL",
                        "start": 27,
                        "end": 34,
                        "source_model": "BC5CDR",
                        "confidence": 0.9455,
                    },
                    {
                        "text": "pneumonia",
                        "label": "DISEASE",
                        "start": 39,
                        "end": 48,
                        "source_model": "BC5CDR, NCBI Disease",
                        "confidence": 0.7249,
                    },
                ],
            }
        }
    )

    text: str = Field(..., description="Original input text.")
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of extracted biomedical entities with exact offsets and confidence.",
    )

    @model_validator(mode="after")
    def validate_entity_offsets_match_text(self) -> "PredictResponse":
        text_len = len(self.text)
        for ent in self.entities:
            if ent.end > text_len:
                raise ValueError(
                    f"Entity end offset ({ent.end}) exceeds total text length ({text_len}) for entity '{ent.text}'."
                )
            actual_slice = self.text[ent.start : ent.end]
            if actual_slice != ent.text:
                raise ValueError(
                    f"Offset mismatch: text[{ent.start}:{ent.end}] is '{actual_slice}', expected '{ent.text}'."
                )
        return self


class HealthResponse(BaseModel):
    """Health check response reflecting actual loaded model status."""
    status: str = Field("healthy", description="API health status.")
    models_loaded: bool = Field(..., description="True if all models are loaded in memory.")
    models: List[str] = Field(..., description="List of initialized model names.")


class RootResponse(BaseModel):
    """Root endpoint basic metadata."""
    name: str = Field("Trievo Clinical NER API", description="Service title.")
    status: str = Field("running", description="Operational status.")
    version: str = Field("1.0.0", description="Semantic version.")


class ErrorResponse(BaseModel):
    """Structured error message."""
    detail: str = Field(..., description="Human-readable explanation of error.")
