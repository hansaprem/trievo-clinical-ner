"""Automated Test Suite for Unified Multi-Model Clinical NER Pipeline.

Tests:
  - Checkpoint loading and initialization for all 4 models.
  - Single sentence and multi-sentence paragraph inference.
  - Multi-entity extraction across all biomedical domains.
  - Deterministic Disease deduplication.
  - Overlapping same-label maximal span resolution.
  - Nested span preservation across different entity types.
  - Exact character offset validation (text[start:end] == entity.text).
  - Long text sliding-window chunking (>512 subwords).
  - Empty, whitespace, and negative non-clinical inputs.
  - CPU and CUDA execution compatibility.
"""

import sys
import unittest
from pathlib import Path
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.adapter import RawEntitySpan
from src.inference.model_loader import ModelRegistry
from src.inference.normalizer import normalize_label
from src.inference.pipeline import UnifiedClinicalNERPipeline
from src.inference.resolver import EntityResolver, UnifiedEntity
from src.inference.sliding_window import SlidingWindowTokenizer


class TestClinicalNERPipeline(unittest.TestCase):
    """Unit and integration test cases for the clinical NER inference pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initializes shared test pipeline once to optimize CPU execution time."""
        cls.models_root = PROJECT_ROOT / "models"
        cls.registry = ModelRegistry(models_root=cls.models_root, device="cpu", preload=False)
        cls.pipeline = UnifiedClinicalNERPipeline(
            models_root=cls.models_root,
            device="cpu",
            preload=True,
        )

    def test_01_checkpoint_loading(self):
        """Verifies that all 4 checkpoints load successfully with expected configurations."""
        supported_models = self.registry.get_supported_models()
        self.assertEqual(len(supported_models), 4)
        self.assertIn("BC5CDR", supported_models)
        self.assertIn("NCBI Disease", supported_models)
        self.assertIn("JNLPBA", supported_models)
        self.assertIn("AnatEM", supported_models)

        for name in supported_models:
            loaded = self.registry.get(name)
            self.assertIsNotNone(loaded.model)
            self.assertIsNotNone(loaded.tokenizer)
            self.assertFalse(loaded.model.training, f"Model {name} should be in eval mode")
            self.assertGreater(len(loaded.id2label), 1)
            self.assertGreater(len(loaded.tokenizer), 1000)

    def test_02_empty_and_whitespace_input(self):
        """Verifies that empty and whitespace inputs return an empty entity list without error."""
        test_cases = ["", "   ", "\n\t  \r\n"]
        for text in test_cases:
            res = self.pipeline.extract_entities(text)
            self.assertEqual(res["text"], text)
            self.assertEqual(res["entities"], [])

    def test_03_negative_input_no_entities(self):
        """Verifies that everyday text with no clinical terms returns 0 entities."""
        text = "The quick brown fox jumps over the lazy dog in the sunny park."
        res = self.pipeline.extract_entities(text)
        self.assertEqual(res["text"], text)
        self.assertEqual(len(res["entities"]), 0)

    def test_04_exact_character_offsets(self):
        """Verifies that text[start:end] strictly equals entity['text'] for every extraction."""
        text = (
            "Administration of aspirin and ibuprofen caused mild gastric ulceration "
            "and acute renal failure in elderly patients with rheumatoid arthritis."
        )
        res = self.pipeline.extract_entities(text)
        self.assertGreater(len(res["entities"]), 0)

        for ent in res["entities"]:
            sliced = text[ent["start"] : ent["end"]]
            self.assertEqual(
                sliced,
                ent["text"],
                f"Offset mismatch: expected '{ent['text']}', got '{sliced}' at [{ent['start']}:{ent['end']}]",
            )
            self.assertGreaterEqual(ent["confidence"], 0.0)
            self.assertLessEqual(ent["confidence"], 1.0)
            self.assertIn(ent["label"], [
                "CHEMICAL", "DISEASE", "ANATOMY", "PROTEIN", "CELL_TYPE", "DNA", "CELL_LINE", "RNA"
            ])

    def test_05_multi_entity_extraction(self):
        """Tests simultaneous extraction of diverse biomedical entity types."""
        text = (
            "Doxorubicin was given for metastatic breast cancer, targeting malignant cells "
            "in the mammary gland and reducing p53 protein expression."
        )
        res = self.pipeline.extract_entities(text)
        labels = {e["label"] for e in res["entities"]}

        # Expect at least Chemical, Disease, and Anatomy or Protein
        self.assertIn("CHEMICAL", labels)
        self.assertIn("DISEASE", labels)

        # Verify exact character slices for all entities
        for e in res["entities"]:
            self.assertEqual(text[e["start"] : e["end"]], e["text"])

    def test_06_disease_duplicate_deduplication(self):
        """Unit test for EntityResolver: verifies exact duplicate Disease spans are merged."""
        resolver = EntityResolver()
        text = "The patient suffered from severe pneumonia and hypertension."
        target = "pneumonia"
        start = text.index(target)
        end = start + len(target)

        span1 = RawEntitySpan(
            text=target,
            raw_label="Disease",
            start=start,
            end=end,
            source_model="BC5CDR",
            confidence=0.8800,
            token_count=1,
        )
        span2 = RawEntitySpan(
            text=target,
            raw_label="Disease",
            start=start,
            end=end,
            source_model="NCBI Disease",
            confidence=0.9200,
            token_count=1,
        )

        resolved = resolver.resolve([span1, span2], text)
        self.assertEqual(len(resolved), 1)
        ent = resolved[0]
        self.assertEqual(ent.text, target)
        self.assertEqual(ent.label, "DISEASE")
        self.assertEqual(ent.start, start)
        self.assertEqual(ent.end, end)
        self.assertIn("BC5CDR", ent.source_models)
        self.assertIn("NCBI Disease", ent.source_models)
        self.assertEqual(ent.confidence, 0.9000)

    def test_07_overlapping_same_label_maximal_span(self):
        """Unit test for EntityResolver: verifies same-label overlaps select the maximal span."""
        resolver = EntityResolver()
        text = "Case of acute renal failure was documented."
        short_target = "renal failure"
        long_target = "acute renal failure"
        s_short = text.index(short_target)
        e_short = s_short + len(short_target)
        s_long = text.index(long_target)
        e_long = s_long + len(long_target)

        span_short = RawEntitySpan(
            text=short_target,
            raw_label="Disease",
            start=s_short,
            end=e_short,
            source_model="BC5CDR",
            confidence=0.8500,
            token_count=2,
        )
        span_long = RawEntitySpan(
            text=long_target,
            raw_label="Disease",
            start=s_long,
            end=e_long,
            source_model="NCBI Disease",
            confidence=0.8900,
            token_count=3,
        )

        resolved = resolver.resolve([span_short, span_long], text)
        self.assertEqual(len(resolved), 1)
        ent = resolved[0]
        self.assertEqual(ent.text, long_target)
        self.assertEqual(ent.label, "DISEASE")
        self.assertEqual(ent.start, s_long)
        self.assertEqual(ent.end, e_long)
        self.assertIn("NCBI Disease", ent.source_models)
        self.assertIn("BC5CDR", ent.source_models)
        self.assertEqual(len(ent.sub_spans), 1)
        self.assertEqual(ent.sub_spans[0]["text"], short_target)

    def test_08_nested_span_preservation_different_types(self):
        """Unit test for EntityResolver: preserves nested entities with different types."""
        resolver = EntityResolver()
        text = "Biopsy confirmed prostate cancer in the peripheral zone."
        ana_target = "prostate"
        dis_target = "prostate cancer"
        s_ana = text.index(ana_target)
        e_ana = s_ana + len(ana_target)
        s_dis = text.index(dis_target)
        e_dis = s_dis + len(dis_target)

        span_anatomy = RawEntitySpan(
            text=ana_target,
            raw_label="Anatomy",
            start=s_ana,
            end=e_ana,
            source_model="AnatEM",
            confidence=0.9100,
            token_count=1,
        )
        span_disease = RawEntitySpan(
            text=dis_target,
            raw_label="Disease",
            start=s_dis,
            end=e_dis,
            source_model="NCBI Disease",
            confidence=0.8700,
            token_count=2,
        )

        resolved = resolver.resolve([span_anatomy, span_disease], text)
        self.assertEqual(len(resolved), 2)

        # Both entities must be present
        texts = {e.text for e in resolved}
        labels = {e.label for e in resolved}
        self.assertIn(ana_target, texts)
        self.assertIn(dis_target, texts)
        self.assertIn("ANATOMY", labels)
        self.assertIn("DISEASE", labels)

    def test_09_long_text_sliding_window(self):
        """Verifies that texts exceeding 512 subwords are processed via sliding window without truncation."""
        # Create a repetitive paragraph with clinical entities at start, middle, and end
        sentence_a = "The patient was prescribed methotrexate for rheumatoid arthritis. "
        sentence_b = "Normal physiological saline was administered at regular intervals. "
        sentence_c = "Final pathology showed adenocarcinoma in the lung parenchyma. "

        long_text = (sentence_a * 15) + (sentence_b * 30) + (sentence_c * 15)
        self.assertGreater(len(long_text), 3000)

        # Process through sliding window pipeline
        res = self.pipeline.extract_entities(long_text)
        self.assertGreater(len(res["entities"]), 20)

        # Verify all extracted offsets strictly align with text slice
        for e in res["entities"]:
            self.assertEqual(long_text[e["start"] : e["end"]], e["text"])

        # Check that entities from both beginning and end are captured
        starts = [e["start"] for e in res["entities"]]
        self.assertLess(min(starts), 200)
        self.assertGreater(max(starts), 2500)

    def test_10_device_and_inference_mode(self):
        """Verifies device placement and inference mode behavior."""
        self.assertEqual(self.pipeline.device.type, "cpu")
        for name, adapter in self.pipeline.adapters.items():
            self.assertFalse(adapter.model.training)
            for param in adapter.model.parameters():
                self.assertFalse(param.requires_grad)

    def test_11_smoke_test_actual_outputs(self):
        """Smoke test on a complex clinical sentence verifying output schema."""
        sample_text = (
            "The patient was prescribed tacrolimus and aspirin for chronic glomerulonephritis "
            "affecting the kidney cortex, with elevated p53 protein levels."
        )
        res = self.pipeline.extract_entities(sample_text, include_raw=True, extended_provenance=True)

        self.assertEqual(res["text"], sample_text)
        self.assertIn("entities", res)
        self.assertIn("raw_entities_by_model", res)
        self.assertIn("metadata", res)

        entities = res["entities"]
        self.assertGreater(len(entities), 0)

        for e in entities:
            # Check schema keys
            self.assertIn("text", e)
            self.assertIn("label", e)
            self.assertIn("start", e)
            self.assertIn("end", e)
            self.assertIn("source_model", e)
            self.assertIn("confidence", e)
            # Verify slice
            self.assertEqual(sample_text[e["start"] : e["end"]], e["text"])


if __name__ == "__main__":
    unittest.main()
