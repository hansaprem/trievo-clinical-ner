"""
Production Verification Script for Render Deployed FastAPI Backend.

Executes tests against live endpoints:
- GET /
- GET /health
- GET /docs
- POST /predict with sample clinical sentence
- POST /predict with empty string to verify validation handling

Measures real latency and verifies returned JSON structures.
NEVER includes or expects credentials.
"""

import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_TEST_TEXT = "The patient was treated with aspirin for pneumonia."


def make_request(url: str, method: str = "GET", data: dict = None):
    headers = {"User-Agent": "Trievo-Production-Verifier/1.0"}
    encoded_data = None
    if data is not None:
        encoded_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed = time.perf_counter() - start_time
            body = resp.read().decode("utf-8")
            return resp.status, body, elapsed, None
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start_time
        body = e.read().decode("utf-8")
        return e.code, body, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return 0, "", elapsed, str(e)


def run_verification(base_url: str):
    base_url = base_url.rstrip("/")
    print(f"===========================================================")
    print(f"TRIEVO CLINICAL NER — PRODUCTION BACKEND VERIFICATION")
    print(f"Target URL: {base_url}")
    print(f"===========================================================\n")

    results = {}

    # 1. GET /
    print("[1/5] Testing GET / ...")
    status, body, elapsed, err = make_request(f"{base_url}/")
    print(f"      Status: {status} | Latency: {elapsed*1000:.1f}ms")
    if err:
        print(f"      Error: {err}")
    else:
        print(f"      Response: {body[:150]}")
    results["root"] = {"status": status, "elapsed_ms": round(elapsed * 1000, 1), "body": body}

    # 2. GET /health
    print("\n[2/5] Testing GET /health ...")
    status, body, elapsed, err = make_request(f"{base_url}/health")
    print(f"      Status: {status} | Latency: {elapsed*1000:.1f}ms")
    if err:
        print(f"      Error: {err}")
    else:
        print(f"      Response: {body}")
    results["health"] = {"status": status, "elapsed_ms": round(elapsed * 1000, 1), "body": body}

    # 3. GET /docs
    print("\n[3/5] Testing GET /docs (Swagger UI) ...")
    status, body, elapsed, err = make_request(f"{base_url}/docs")
    print(f"      Status: {status} | Latency: {elapsed*1000:.1f}ms")
    results["docs"] = {"status": status, "elapsed_ms": round(elapsed * 1000, 1)}

    # 4. POST /predict (Clinical sentence)
    print(f"\n[4/5] Testing POST /predict ...")
    payload = {"text": DEFAULT_TEST_TEXT}
    print(f"      Input Text: \"{DEFAULT_TEST_TEXT}\"")
    status, body, elapsed, err = make_request(f"{base_url}/predict", method="POST", data=payload)
    print(f"      Status: {status} | Measured Latency: {elapsed*1000:.1f}ms")
    if status == 200:
        data = json.loads(body)
        entities = data.get("entities", [])
        print(f"      Extracted Entities ({len(entities)} found):")
        for ent in entities:
            print(f"        - [{ent.get('label')}] \"{ent.get('text')}\" (offsets: {ent.get('start')}:{ent.get('end')}, conf: {ent.get('confidence', 0):.3f}, source: {ent.get('source_model')})")
        results["predict_valid"] = {
            "status": status,
            "elapsed_ms": round(elapsed * 1000, 1),
            "entities": entities,
        }
    else:
        print(f"      Failure: {body}")
        results["predict_valid"] = {"status": status, "elapsed_ms": round(elapsed * 1000, 1), "error": body}

    # 5. POST /predict (Empty string validation)
    print("\n[5/5] Testing POST /predict with empty text {'text': ''} ...")
    status, body, elapsed, err = make_request(f"{base_url}/predict", method="POST", data={"text": ""})
    print(f"      Status: {status} (Expected 400 Bad Request) | Latency: {elapsed*1000:.1f}ms")
    print(f"      Response: {body}")
    results["predict_empty"] = {"status": status, "elapsed_ms": round(elapsed * 1000, 1), "body": body}

    print("\n===========================================================")
    print("VERIFICATION COMPLETED")
    print("===========================================================")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_live_render_backend.py <RENDER_URL>")
        sys.exit(1)
    target = sys.argv[1]
    run_verification(target)
