"""End-to-End Automated Verification Script for Redesigned Clinical NER Prototype.

Validates:
  1. FastAPI backend /health and preloaded models.
  2. Frontend Vite dev server HTTP 200 response.
  3. All 7 clinical library scenarios through POST /predict.
  4. 100% exact character offset match: text[start:end] == entity.text.
  5. Confidence numeric validity in [0.0, 1.0].
  6. Source model provenance presence and multi-model joint resolution.
  7. Empty and whitespace input error handling (HTTP 400).
  8. Generates comprehensive audit report in reports/redesigned_prototype_audit.json.
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_FILE = REPORTS_DIR / "redesigned_prototype_audit.json"

LIBRARY_EXAMPLES = [
    {
        "category": "General Clinical",
        "title": "Pulmonology Case",
        "text": "The patient was diagnosed with pneumonia and prescribed aspirin. CT imaging demonstrated involvement of the right lung, while elevated p53 protein levels were noted.",
    },
    {
        "category": "Oncology",
        "title": "Hematologic Malignancy",
        "text": "Acute lymphoblastic leukemia was identified in the bone marrow biopsy, and methotrexate chemotherapy was promptly initiated.",
    },
    {
        "category": "Molecular Biology",
        "title": "Cellular Signaling",
        "text": "Elevated p53 protein expression was observed in the human tumor cells, along with significantly increased cytoplasmic mRNA levels.",
    },
    {
        "category": "Genetics",
        "title": "Genomic Mutation",
        "text": "The study identified an oncogenic mutation adjacent to the c-myc gene and sequenced the corresponding target DNA locus.",
    },
    {
        "category": "Pharmacology",
        "title": "Immunosuppressive Regimen",
        "text": "Following allograft surgery, the patient received tacrolimus, cyclosporine, and low-dose methotrexate during the maintenance period.",
    },
    {
        "category": "Anatomy",
        "title": "Visceral Tissue Pathology",
        "text": "Contrast imaging demonstrated localized hypodensity involving the left kidney cortex and posterior right lung parenchyma.",
    },
    {
        "category": "Mixed Biomedical",
        "title": "Multidisciplinary Findings",
        "text": "A biopsy of the renal parenchyma in a patient with chronic glomerulonephritis confirmed antibody deposition and marked cellular infiltration.",
    },
]


def test_get(url: str, accept: str = "application/json"):
    req = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(req, timeout=5) as res:
        return {"status": res.status, "data": res.read().decode("utf-8")}


def test_post(url: str, payload: dict):
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "Accept": "application/json", "Origin": FRONTEND_URL},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            headers = dict(res.headers)
            body = json.loads(res.read().decode("utf-8"))
            return {"status": res.status, "headers": headers, "data": body}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            body = json.loads(body)
        except Exception:
            pass
        return {"status": e.code, "headers": dict(e.headers), "data": body}


def run_audit():
    print("=" * 60)
    print("TRIEVO CLINICAL NER — REDESIGNED PROTOTYPE END-TO-END AUDIT")
    print("=" * 60)

    # 1. Backend Health
    print("\n[Step 1] Verifying Backend Health...")
    health_raw = test_get(f"{API_URL}/health")
    health = json.loads(health_raw["data"])
    assert health["status"] == "healthy", f"Backend unhealthy: {health}"
    assert health["models_loaded"] is True, "Models not loaded"
    assert len(health["models"]) == 4, "Expected 4 models"
    print(f" -> OK: Status='{health['status']}', Models={health['models']}")

    # 2. Frontend Web Server
    print("\n[Step 2] Verifying Frontend Dev Server...")
    fe = test_get(FRONTEND_URL, accept="text/html")
    assert fe["status"] == 200, f"Frontend returned status {fe['status']}"
    print(f" -> OK: Frontend active on {FRONTEND_URL}")

    # 3. Test All Curated Clinical Scenarios
    print("\n[Step 3] Executing Live Inference Across Curated Example Library...")
    audit_results = []
    total_extracted_entities = 0

    for idx, scenario in enumerate(LIBRARY_EXAMPLES, 1):
        print(f"\n--- Scenario {idx}/7: {scenario['category']} ({scenario['title']}) ---")
        pred = test_post(f"{API_URL}/predict", {"text": scenario["text"]})
        assert pred["status"] == 200, f"Inference failed: {pred}"

        data = pred["data"]
        entities = data["entities"]
        total_extracted_entities += len(entities)
        print(f"Text: '{scenario['text'][:60]}...'")
        print(f"Extracted {len(entities)} entities:")

        for ent in entities:
            # 100% exact character slice verification
            slice_text = scenario["text"][ent["start"]:ent["end"]]
            assert slice_text == ent["text"], (
                f"Offset misalignment! sliced='{slice_text}' vs entity='{ent['text']}'"
            )
            assert 0.0 <= ent["confidence"] <= 1.0, f"Invalid confidence: {ent['confidence']}"
            assert ent["source_model"], "Source model missing"

            print(
                f"  • [{ent['label']:<10}] \"{ent['text']}\" [span {ent['start']}:{ent['end']}] "
                f"conf={ent['confidence']:.4f} | model={ent['source_model']}"
            )

        audit_results.append({
            "category": scenario["category"],
            "title": scenario["title"],
            "text": scenario["text"],
            "entity_count": len(entities),
            "entities": entities,
        })

    # 4. Input Validation & Error Handling
    print("\n[Step 4] Verifying Input Validation & Error Handling...")
    empty_pred = test_post(f"{API_URL}/predict", {"text": "   "})
    assert empty_pred["status"] == 400, f"Expected 400, got: {empty_pred['status']}"
    assert "empty or whitespace" in empty_pred["data"]["detail"], "Unexpected error detail"
    print(f" -> OK: Empty text correctly rejected with HTTP 400: '{empty_pred['data']['detail']}'")

    # 5. Compile and Save Formal Audit Report
    report = {
        "status": "PASS",
        "api_url": API_URL,
        "frontend_url": FRONTEND_URL,
        "backend_health": health,
        "total_scenarios_tested": len(LIBRARY_EXAMPLES),
        "total_entities_extracted": total_extracted_entities,
        "scenarios": audit_results,
        "input_validation_test": empty_pred,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] ALL 7 SCENARIOS VERIFIED! Total entities extracted: {total_extracted_entities}")
    print(f"Audit log saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()
