"""End-to-End Automated Verification Script for Clinical NER System.

Verifies:
  1. Backend API Root and Health endpoints.
  2. Frontend Vite Dev Server availability.
  3. POST /predict with multiple clinical test cases.
  4. Exact character offset alignment for every extracted entity.
  5. Confidence score bounds and source model attribution.
  6. Empty input validation (HTTP 400).
  7. CORS headers.
"""

import json
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_FILE = REPORTS_DIR / "e2e_verification_results.json"


def test_get(url: str, accept: str = "application/json"):
    req = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(req, timeout=5) as res:
        status = res.status
        content_type = res.headers.get("Content-Type", "")
        body = res.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body[:200]
        return {"status": status, "content_type": content_type, "data": parsed}


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


def run_full_verification():
    print("[1/6] Testing Backend Root & Health...")
    root_res = test_get(f"{API_URL}/")
    health_res = test_get(f"{API_URL}/health")
    assert root_res["status"] == 200, f"Root failed: {root_res}"
    assert health_res["status"] == 200, f"Health failed: {health_res}"
    assert health_res["data"]["models_loaded"] is True, "Models not loaded"
    assert len(health_res["data"]["models"]) == 4, "Expected 4 models loaded"
    print(" -> Backend Healthy! Models:", health_res["data"]["models"])

    print("\n[2/6] Testing Frontend Web Server...")
    fe_res = test_get(FRONTEND_URL, accept="text/html")
    assert fe_res["status"] == 200, f"Frontend server failed: {fe_res}"
    print(" -> Frontend Available on", FRONTEND_URL)

    print("\n[3/6] Testing Primary Clinical Example...")
    example_text = (
        "The patient was diagnosed with pneumonia and prescribed aspirin. "
        "CT imaging demonstrated involvement of the right lung, while elevated p53 protein levels were noted."
    )
    pred_res1 = test_post(f"{API_URL}/predict", {"text": example_text})
    assert pred_res1["status"] == 200, f"Prediction 1 failed: {pred_res1}"
    entities1 = pred_res1["data"]["entities"]
    print(f" -> Extracted {len(entities1)} entities:")
    for ent in entities1:
        # Verify exact character offset alignment
        sliced = example_text[ent["start"]:ent["end"]]
        assert sliced == ent["text"], f"Offset mismatch! '{sliced}' vs '{ent['text']}'"
        assert 0.0 <= ent["confidence"] <= 1.0, f"Invalid confidence: {ent['confidence']}"
        print(f"    • [{ent['label']:<10}] '{ent['text']}' [span {ent['start']}:{ent['end']}] "
              f"conf={ent['confidence']:.4f} source={ent['source_model']}")

    print("\n[4/6] Testing Secondary Oncology / Pharmacology Text...")
    example_text_2 = (
        "Acute lymphoblastic leukemia was identified in bone marrow biopsy, treated with methotrexate."
    )
    pred_res2 = test_post(f"{API_URL}/predict", {"text": example_text_2})
    assert pred_res2["status"] == 200, f"Prediction 2 failed: {pred_res2}"
    entities2 = pred_res2["data"]["entities"]
    print(f" -> Extracted {len(entities2)} entities:")
    for ent in entities2:
        sliced = example_text_2[ent["start"]:ent["end"]]
        assert sliced == ent["text"], f"Offset mismatch! '{sliced}' vs '{ent['text']}'"
        print(f"    • [{ent['label']:<10}] '{ent['text']}' [span {ent['start']}:{ent['end']}] "
              f"conf={ent['confidence']:.4f} source={ent['source_model']}")

    print("\n[5/6] Testing Input Validation & Error Handling...")
    empty_res = test_post(f"{API_URL}/predict", {"text": "   "})
    assert empty_res["status"] == 400, f"Expected 400 for empty text, got: {empty_res['status']}"
    print(" -> Empty text properly returned HTTP 400:", empty_res["data"])

    print("\n[6/6] Testing CORS Access-Control Headers...")
    cors_origin = pred_res1["headers"].get("access-control-allow-origin") or pred_res1["headers"].get("Access-Control-Allow-Origin")
    print(" -> Access-Control-Allow-Origin:", cors_origin)
    assert cors_origin is not None, "CORS header missing!"

    # Compile report output
    report_data = {
        "status": "SUCCESS",
        "api_url": API_URL,
        "frontend_url": FRONTEND_URL,
        "backend_health": health_res["data"],
        "example_1": {
            "input_text": example_text,
            "entities_count": len(entities1),
            "entities": entities1,
        },
        "example_2": {
            "input_text": example_text_2,
            "entities_count": len(entities2),
            "entities": entities2,
        },
        "empty_text_validation": empty_res,
        "cors_origin_header": cors_origin,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] All end-to-end verifications passed! Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_full_verification()
