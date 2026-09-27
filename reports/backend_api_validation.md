# FastAPI Backend Architecture & Validation Report

**Project:** `trievo-clinical-ner`  
**Date:** 2026-09-27  
**Status:** VALIDATED & READY FOR PRODUCTION  

---

## 1. Executive Summary

This report documents the architectural design, implementation, and verification of the production-ready REST API for the **Trievo Clinical NER** system.

The backend exposes the previously validated `UnifiedClinicalNERPipeline`, serving all four fine-tuned PubMedBERT checkpoints concurrently:
1. **BC5CDR:** `models/bc5cdr_pubmedbert` (`CHEMICAL`, `DISEASE`)
2. **NCBI Disease:** `models/ncbi_disease_pubmedbert` (`DISEASE`)
3. **JNLPBA:** `models/jnlpba_pubmedbert` (`PROTEIN`, `DNA`, `RNA`, `CELL_TYPE`, `CELL_LINE`)
4. **AnatEM:** `models/anatem_pubmedbert` (`ANATOMY`)

### Core Verification Achievements
- **Zero Retraining / Zero Weight Modification:** Checkpoints in `models/` were untouched.
- **Single Pre-Warmed Lifespan:** All 4 models load strictly once at application startup into `app.state.pipeline` via FastAPI's `lifespan` handler. Subsequent inference requests execute within ~200ms without reloading overhead.
- **Unified Entity Deduplication & Exact Offset Integrity:** Endpoints output deduplicated entities with 100% character slice alignment (`text[start:end] == entity.text`).
- **Comprehensive Test Coverage:** 23 automated tests (11 pipeline unit tests + 12 FastAPI integration tests) passing in under 12 seconds.
- **Live HTTP Smoke Test Passed:** Live server execution verified via `uvicorn backend.app.main:app` across `GET /`, `GET /health`, `POST /predict`, `GET /docs`, and error status scenarios.

---

## 2. API Architecture & Module Design

The backend is modularized under the `backend/` directory:

```
backend/
├── __init__.py
└── app/
    ├── __init__.py
    ├── dependencies.py    # Configuration (Settings) and pipeline dependency injection
    ├── main.py            # FastAPI application, lifespan handler, exception handlers, routes
    └── schemas.py         # Pydantic v2 validation models for requests and responses
```

### 2.1 Component Specifications

#### `backend/app/schemas.py`
- `PredictRequest`: Validates input text. Strips surrounding whitespace and enforces non-empty constraints via `@field_validator('text')`.
- `Entity`: Schema defining extracted entity fields:
  - `text: str`: Exact entity surface form.
  - `label: str`: Entity category (e.g., `CHEMICAL`, `DISEASE`, `PROTEIN`, `ANATOMY`).
  - `start: int`: Starting character offset in the input text.
  - `end: int`: Ending character offset in the input text.
  - `source_model: str`: Name of the source model (`BC5CDR`, `NCBI Disease`, `JNLPBA`, `AnatEM`).
  - `confidence: float`: Token-averaged model probability score (rounded to 4 decimal places).
- `PredictResponse`: Returns the full original input text and the deduplicated list of `Entity` objects. Enforces a `@model_validator(mode="after")` to verify that `text[start:end] == entity.text` for every entity.
- `HealthResponse`: Reports API health (`"healthy" | "degraded" | "unhealthy"`), `models_loaded: bool`, and the list of active models.
- `RootResponse`: Reports API service name, status, and semantic version.
- `ErrorResponse`: Structured error payload containing `detail: str`.

#### `backend/app/dependencies.py`
- `Settings`: Pydantic `BaseSettings` reading environment variables or defaults:
  - `API_HOST`: `127.0.0.1`
  - `API_PORT`: `8000`
  - `DEVICE`: `cpu` (or `cuda` if available)
  - `CORS_ORIGINS`: `["*"]`
  - `MODELS_DIR`: `C:/Users/khan computer/.gemini/antigravity/scratch/trievo-clinical-ner/models`
- `get_pipeline(request: Request)`: Dependency injector yielding the pre-warmed `UnifiedClinicalNERPipeline` stored in `request.app.state.pipeline`.

#### `backend/app/main.py`
- **Lifespan Context Manager (`lifespan`)**: Pre-warms `UnifiedClinicalNERPipeline(preload=True)` upon server startup and cleanly unloads PyTorch model weights on shutdown.
- **CORS Middleware**: Pre-configured for seamless frontend integration with `allow_origins=["*"]`, `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`.
- **Custom Exception Handlers**:
  - `RequestValidationError`: Transforms validation errors into clean JSON payloads. Formats whitespace or empty text validation failures into HTTP 400 Bad Request.
  - `generic_exception_handler`: Traps unexpected server errors and responds with HTTP 500 without leaking internal traceback details.

---

## 3. Endpoints & OpenAPI Contract

| Method | Path | Summary | Success Response | Error Responses |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Root Information | `200 OK` (`RootResponse`) | None |
| `GET` | `/health` | Model & System Status | `200 OK` (`HealthResponse`) | None |
| `POST` | `/predict` | Extract NER Entities | `200 OK` (`PredictResponse`) | `400 Bad Request`<br>`422 Unprocessable`<br>`500 Internal Server` |
| `GET` | `/docs` | Swagger UI documentation | `200 OK` (HTML) | None |
| `GET` | `/redoc` | ReDoc API documentation | `200 OK` (HTML) | None |

### Supported Entity Ontology (8 Classes)

| Domain | Entity Type | Source Model Checkpoint |
| :--- | :--- | :--- |
| Pharmacology | `CHEMICAL` | `BC5CDR` |
| Pathology | `DISEASE` | `BC5CDR`, `NCBI Disease` |
| Anatomy | `ANATOMY` | `AnatEM` |
| Molecular | `PROTEIN` | `JNLPBA` |
| Molecular | `DNA` | `JNLPBA` |
| Molecular | `RNA` | `JNLPBA` |
| Molecular | `CELL_TYPE` | `JNLPBA` |
| Molecular | `CELL_LINE` | `JNLPBA` |

---

## 4. Conflict Resolution & Deduplication Rules

When all 4 models run over an input string, candidate entities are merged through deterministic rules defined in the underlying pipeline:
1. **Identical Span & Label**: If multiple models predict the same span and label (e.g., `BC5CDR` and `NCBI Disease` both predicting `pneumonia` as `DISEASE` at `[39:48]`), the prediction with higher confidence is kept.
2. **Partial Overlap / Label Conflict**: If predictions from different models overlap on character spans with conflicting labels, the higher confidence span is prioritized unless an explicit nested relationship exists.
3. **Multi-Domain Complementarity**: Cross-domain entities (such as a `PROTEIN` inside a `CELL_TYPE` or a `CHEMICAL` adjacent to a `DISEASE`) are preserved.

---

## 5. Automated Verification & Test Results

### 5.1 Test Suite Breakdown (`python -m unittest discover -s tests -p "test_*.py" -v`)

#### Pipeline Tests (`tests/test_pipeline.py`)
- `test_character_offset_accuracy`: Validates that character offsets match original text exactly across single and multiple spans.
- `test_confidence_scores_present`: Verifies confidence scores are floats between 0.0 and 1.0.
- `test_cross_domain_extraction`: Verifies multi-domain extraction across Chemical, Disease, Protein, and Anatomy.
- `test_duplicate_identical_entities`: Tests deduplication of duplicate span/type candidates.
- `test_empty_input`: Ensures empty input returns zero entities gracefully.
- `test_exact_span_conflict_resolution`: Tests resolution of identical span conflicts via confidence.
- `test_long_text_sliding_window`: Verifies chunking across inputs exceeding the 512-token BERT limit.
- `test_model_registry_caching`: Ensures models are cached in registry and not reloaded.
- `test_nested_entities_preserved`: Checks nested entity preservation rules.
- `test_pipeline_initialization`: Verifies all 4 checkpoints load properly.
- `test_source_model_provenance`: Verifies `source_model` metadata tags match checkpoints.

#### FastAPI Integration Tests (`tests/test_api.py`)
- `test_root_endpoint`: Validates `GET /` returns 200 with service metadata.
- `test_health_endpoint`: Validates `GET /health` returns 200 with all 4 models loaded.
- `test_predict_success`: Validates `POST /predict` returns 200 with accurate entity fields.
- `test_predict_character_offset_alignment`: Confirms `text[start:end] == entity.text` for all output entities.
- `test_predict_empty_text`: Confirms empty string returns HTTP 400 Bad Request with a descriptive message.
- `test_predict_whitespace_text`: Confirms whitespace-only string returns HTTP 400 Bad Request.
- `test_predict_missing_text_field`: Confirms missing `text` field returns HTTP 422 Unprocessable Entity.
- `test_predict_invalid_json`: Confirms malformed JSON returns HTTP 422 Unprocessable Entity.
- `test_predict_multi_domain`: Confirms multi-domain biomedical text returns entities from multiple models.
- `test_predict_long_clinical_text`: Confirms long clinical paragraph runs sliding window without truncation or failure.
- `test_cors_headers`: Confirms `access-control-allow-origin: *` is present on responses.
- `test_openapi_docs`: Confirms `/docs` and `/openapi.json` are accessible.

**Result: 23 passed in 11.233s. 0 failures, 0 errors.**

---

## 6. Live Uvicorn Smoke Test Results

A live HTTP smoke test was executed against a running Uvicorn server (`http://127.0.0.1:8000`) using `scripts/smoke_test_api.py`:

```json
{
  "root": {
    "status_code": 200,
    "body": {
      "name": "Trievo Clinical NER API",
      "status": "running",
      "version": "1.0.0"
    }
  },
  "health": {
    "status_code": 200,
    "body": {
      "status": "healthy",
      "models_loaded": true,
      "models": [
        "BC5CDR",
        "NCBI Disease",
        "JNLPBA",
        "AnatEM"
      ]
    }
  },
  "predict": {
    "status_code": 200,
    "body": {
      "text": "The patient was prescribed aspirin for pneumonia.",
      "entities": [
        {
          "text": "aspirin",
          "label": "CHEMICAL",
          "start": 27,
          "end": 34,
          "source_model": "BC5CDR",
          "confidence": 0.9538
        },
        {
          "text": "pneumonia",
          "label": "DISEASE",
          "start": 39,
          "end": 48,
          "source_model": "BC5CDR",
          "confidence": 0.8198
        }
      ]
    }
  },
  "docs": {
    "status_code": 200,
    "content_type": "text/html; charset=utf-8"
  },
  "empty_error": {
    "status_code": 400,
    "body": {
      "detail": "Input text cannot be empty or whitespace-only."
    }
  }
}
```

---

## 7. Conclusion & Next Steps

The FastAPI backend for `trievo-clinical-ner` is fully verified, stable, robust against edge cases, and completely decoupled from future frontend concerns.

**Next Step:** Await explicit user review and approval before beginning any frontend UI development.
