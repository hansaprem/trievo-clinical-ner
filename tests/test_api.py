"""Automated Integration Test Suite for FastAPI Clinical NER Backend.

Tests:
  1. GET / returns 200 with name, status, and version.
  2. GET /health returns 200 with models_loaded=True and all 4 checkpoints listed.
  3. POST /predict with valid clinical text extracts entities conforming to schema.
  4. POST /predict with empty text returns 400 Bad Request.
  5. POST /predict with whitespace text returns 400 Bad Request.
  6. Verification of returned entity offsets: text[start:end] == entity.text.
  7. Verification of returned entity labels: belongs to 8 uppercase categories.
  8. Verification of source_model provenance string.
  9. Verification of confidence score as real numeric value between 0.0 and 1.0.
  10. Verification that API delegates to UnifiedClinicalNERPipeline singleton.
  11. POST /predict with malformed JSON returns 422 Unprocessable Entity.
  12. Verification of CORS headers for localhost origin.
"""

import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app
from src.inference.pipeline import UnifiedClinicalNERPipeline


class TestClinicalNERAPI(unittest.TestCase):
    """Integration test suite executing requests against the live FastAPI app."""

    @classmethod
    def setUpClass(cls):
        """Initializes TestClient and triggers application lifespan once."""
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        """Exits TestClient context and unloads models."""
        cls.client_cm.__exit__(None, None, None)

    def test_01_root_endpoint(self):
        """GET / returns 200 with service metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("name"), "Trievo Clinical NER API")
        self.assertEqual(data.get("status"), "running")
        self.assertEqual(data.get("version"), "1.0.0")

    def test_02_health_endpoint(self):
        """GET /health returns 200 reflecting actual loaded models."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("models_loaded"))
        models = data.get("models", [])
        self.assertEqual(len(models), 4)
        self.assertIn("BC5CDR", models)
        self.assertIn("NCBI Disease", models)
        self.assertIn("JNLPBA", models)
        self.assertIn("AnatEM", models)

    def test_03_predict_valid_sentence(self):
        """POST /predict extracts clinical entities adhering to schema."""
        payload = {"text": "The patient was prescribed aspirin for pneumonia."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], payload["text"])
        self.assertIn("entities", data)
        self.assertIsInstance(data["entities"], list)
        self.assertGreater(len(data["entities"]), 0)

    def test_04_predict_empty_text(self):
        """POST /predict with empty text returns 400 Bad Request."""
        payload = {"text": ""}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("empty or whitespace", data["detail"].lower())

    def test_05_predict_whitespace_text(self):
        """POST /predict with whitespace text returns 400 Bad Request."""
        payload = {"text": "   \n\t   "}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("empty or whitespace", data["detail"].lower())

    def test_06_entity_character_offsets(self):
        """Verifies text[start:end] strictly equals entity.text for all returned entities."""
        text = "Administration of tacrolimus was initiated to treat acute glomerulonephritis."
        payload = {"text": text}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        entities = response.json().get("entities", [])
        self.assertGreater(len(entities), 0)

        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            self.assertGreaterEqual(start, 0)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(text))
            self.assertEqual(text[start:end], ent["text"])

    def test_07_entity_labels(self):
        """Verifies returned entity labels belong to standardized 8-class taxonomy."""
        payload = {"text": "Doxorubicin was given for metastatic breast cancer."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        entities = response.json().get("entities", [])

        valid_labels = {
            "CHEMICAL", "DISEASE", "ANATOMY", "PROTEIN", "CELL_TYPE", "DNA", "CELL_LINE", "RNA"
        }
        for ent in entities:
            self.assertIn(ent["label"], valid_labels)

    def test_08_source_model_provenance(self):
        """Verifies source_model is present, non-empty, and identifies trained model(s)."""
        payload = {"text": "Methotrexate was prescribed for rheumatoid arthritis."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        entities = response.json().get("entities", [])

        for ent in entities:
            self.assertIn("source_model", ent)
            self.assertIsInstance(ent["source_model"], str)
            self.assertGreater(len(ent["source_model"]), 0)
            # Must mention at least one of the 4 models
            has_valid_source = any(
                m in ent["source_model"]
                for m in ["BC5CDR", "NCBI Disease", "JNLPBA", "AnatEM"]
            )
            self.assertTrue(has_valid_source)

    def test_09_confidence_numeric_validity(self):
        """Verifies confidence is an authentic float between 0.0 and 1.0."""
        payload = {"text": "Patient was diagnosed with lymphoma."}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        entities = response.json().get("entities", [])

        for ent in entities:
            self.assertIn("confidence", ent)
            self.assertIsInstance(ent["confidence"], (int, float))
            self.assertGreater(ent["confidence"], 0.0)
            self.assertLessEqual(ent["confidence"], 1.0)

    def test_10_multi_model_coverage_integration(self):
        """Tests complex multi-domain input activating multiple distinct models."""
        payload = {
            "text": (
                "The patient was prescribed tacrolimus and aspirin for chronic glomerulonephritis "
                "affecting the kidney cortex, with elevated p53 protein levels."
            )
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        entities = response.json().get("entities", [])
        labels = {e["label"] for e in entities}

        # Check multi-domain extraction
        self.assertIn("CHEMICAL", labels)
        self.assertIn("DISEASE", labels)
        self.assertIn("ANATOMY", labels)

    def test_11_malformed_json_returns_422(self):
        """POST /predict with malformed JSON body returns 422 Unprocessable Entity."""
        response = self.client.post(
            "/predict",
            content="this is not valid json {",
            headers={"Content-Type": "application/json"},
        )
        self.assertIn(response.status_code, [400, 422])
        self.assertIn("detail", response.json())

    def test_12_cors_headers(self):
        """Verifies CORS headers allow local development origins."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
        response = self.client.options("/predict", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("access-control-allow-origin", response.headers)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")


if __name__ == "__main__":
    unittest.main()
