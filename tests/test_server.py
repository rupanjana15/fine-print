import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from fineprint.inference.server import Worker, make_handler

TEXT = ("You agree to resolve disputes by binding arbitration and waive any class action. " * 3
        + "We may change these terms at any time without notice. " * 3)


class StubModel:
    def analyse(self, chunk: str) -> dict:
        flags = []
        if "arbitration" in chunk:
            flags.append({"category": "dispute_resolution", "severity": "bad",
                          "explanation": "Disputes go to arbitration",
                          "quote": "You agree to resolve disputes by binding arbitration"})
        return {"output": json.dumps({"flags": flags}), "p_empty": 0.1 if flags else 0.99}


@pytest.fixture
def server():
    worker = Worker(None)
    worker.model = StubModel()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(worker))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(url: str, method: str = "GET", body: dict | None = None, origin: str | None = None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Origin": origin} if origin else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}"), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


def test_analyse_job_runs_to_a_graded_result(server):
    ext = "chrome-extension://abc"
    code, body, headers = call(f"{server}/analyse", "POST", {"text": TEXT, "url": "https://x"}, origin=ext)
    assert code == 202 and body["n_chunks"] >= 1
    assert headers["Access-Control-Allow-Origin"] == ext
    for _ in range(100):
        _, job, _ = call(f"{server}/jobs/{body['job']}")
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["done"] == job["total"]
    flags = job["result"]["flags"]
    assert [f["category"] for f in flags] == ["dispute_resolution"] and flags[0]["quote_found"]
    assert job["result"]["grade"] in "ABCDE"


def test_web_pages_are_refused(server):
    code, _, _ = call(f"{server}/analyse", "POST", {"text": TEXT}, origin="https://evil.example")
    assert code == 403
    assert call(f"{server}/health", origin="https://evil.example")[0] == 403


def test_bad_requests(server):
    assert call(f"{server}/analyse", "POST", {"text": "  "})[0] == 400
    assert call(f"{server}/jobs/999")[0] == 404
    assert call(f"{server}/health")[1]["loaded"] is True
