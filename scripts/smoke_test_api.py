"""Live Server Smoke Test for Trievo Clinical NER FastAPI Application.

Starts uvicorn backend.app.main:app on 127.0.0.1:8000,
executes HTTP requests against GET /, GET /health, POST /predict, and GET /docs,
verifies status codes and response schemas, records outputs, and stops the server.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_live_smoke_test():
    host = "127.0.0.1"
    port = 8000
    base_url = f"http://{host}:{port}"

    print(f"Starting uvicorn server on {base_url}...", flush=True)
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for server to warm up (loading 4 models takes ~2.5s)
        client = httpx.Client(base_url=base_url, timeout=30.0)
        max_attempts = 20
        ready = False

        print("Waiting for API server to become ready...", flush=True)
        for i in range(max_attempts):
            try:
                r = client.get("/")
                if r.status_code == 200:
                    ready = True
                    print(f"Server is ready after {(i+1)*1.0:.1f} seconds!", flush=True)
                    break
            except Exception:
                time.sleep(1.0)

        if not ready:
            print("Server failed to respond within timeout.", file=sys.stderr)
            stdout, stderr = server_process.communicate(timeout=5)
            print("Server stdout:", stdout, file=sys.stderr)
            print("Server stderr:", stderr, file=sys.stderr)
            sys.exit(1)

        results = {}

        # 1. Test GET /
        print("\n--- 1. Testing GET / ---", flush=True)
        resp_root = client.get("/")
        print(f"Status Code: {resp_root.status_code}")
        root_json = resp_root.json()
        print(json.dumps(root_json, indent=2))
        assert resp_root.status_code == 200
        assert root_json["status"] == "running"
        results["root"] = {"status_code": resp_root.status_code, "body": root_json}

        # 2. Test GET /health
        print("\n--- 2. Testing GET /health ---", flush=True)
        resp_health = client.get("/health")
        print(f"Status Code: {resp_health.status_code}")
        health_json = resp_health.json()
        print(json.dumps(health_json, indent=2))
        assert resp_health.status_code == 200
        assert health_json["models_loaded"] is True
        assert len(health_json["models"]) == 4
        results["health"] = {"status_code": resp_health.status_code, "body": health_json}

        # 3. Test POST /predict with real clinical sentence
        clinical_sentence = "The patient was prescribed aspirin for pneumonia."
        print(f"\n--- 3. Testing POST /predict ---", flush=True)
        print(f"Input: \"{clinical_sentence}\"")
        resp_predict = client.post("/predict", json={"text": clinical_sentence})
        print(f"Status Code: {resp_predict.status_code}")
        predict_json = resp_predict.json()
        print(json.dumps(predict_json, indent=2))
        assert resp_predict.status_code == 200
        assert predict_json["text"] == clinical_sentence
        assert len(predict_json["entities"]) >= 2
        results["predict"] = {"status_code": resp_predict.status_code, "body": predict_json}

        # 4. Test GET /docs
        print("\n--- 4. Testing GET /docs ---", flush=True)
        resp_docs = client.get("/docs")
        print(f"Status Code: {resp_docs.status_code}")
        assert resp_docs.status_code == 200
        assert "swagger" in resp_docs.text.lower() or "openapi" in resp_docs.text.lower()
        results["docs"] = {"status_code": resp_docs.status_code, "content_type": resp_docs.headers.get("content-type")}

        # 5. Test Error Handling: empty string
        print("\n--- 5. Testing POST /predict with empty text (Error Handling) ---", flush=True)
        resp_empty = client.post("/predict", json={"text": ""})
        print(f"Status Code: {resp_empty.status_code}")
        empty_json = resp_empty.json()
        print(json.dumps(empty_json, indent=2))
        assert resp_empty.status_code == 400
        results["empty_error"] = {"status_code": resp_empty.status_code, "body": empty_json}

        print("\nAll live HTTP endpoint smoke tests PASSED successfully!", flush=True)

        # Save live test results for inclusion in validation report
        output_file = PROJECT_ROOT / "reports" / "live_api_smoke_test.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved live API smoke test records to: {output_file}", flush=True)

    finally:
        print("\nTerminating uvicorn server process...", flush=True)
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("Server stopped cleanly.", flush=True)


if __name__ == "__main__":
    run_live_smoke_test()
