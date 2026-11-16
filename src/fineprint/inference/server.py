"""Local HTTP server for the browser extension (decision D46).

The model runs on your own machine; the extension sends it the page text and shows the flags on the page.
Standard library only, bound to 127.0.0.1, one job at a time (llama.cpp uses all CPU threads anyway).

    uv run --extra eval python -m fineprint.inference.server [--model path/to/fine-print-v2-q4_k_m.gguf]

API (JSON):
    GET    /health          → {"ok": true, "model": name, "loaded": bool, "queue": n}
    POST   /analyse         {"text": "...", "url": "..."} → {"job": id, "n_chunks": n}
    GET    /jobs/<id>       → {"status": queued|running|done|error|cancelled, "done": k, "total": n, "result"?: ...}
    DELETE /jobs/<id>       → cancel

Only requests from a browser extension (Origin chrome-extension:// or moz-extension://) or with no Origin (curl)
are served, so an ordinary web page can't drive your CPU through this port.
"""

import argparse
import atexit
import itertools
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fineprint.chunk import chunk_text
from fineprint.config import load_config
from fineprint.data.clean import clean_text
from fineprint.inference.pipeline import analyse_document

MODEL_REPO = "rupp15/fine-print-gguf"
MODEL_FILE = "fine-print-v2-q4_k_m.gguf"
MAX_CHARS = 150_000  # ~250 chunks, ~10–20 min on an M1; longer pages are almost always not one ToS
EXTENSION_ORIGINS = ("chrome-extension://", "moz-extension://")
KEEP_JOBS = 20


class Cancelled(Exception):
    pass


class Worker:
    """Runs analysis jobs one after another on a lazily loaded model."""

    def __init__(self, model_path: str | None) -> None:
        self.model_path = model_path
        self.model = None
        self.jobs: dict[str, dict] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _load(self):
        if self.model is None:
            from fineprint.inference.gguf import GGUFModel

            path = self.model_path or os.environ.get("FINEPRINT_MODEL")
            if not path:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(MODEL_REPO, MODEL_FILE)
            self.model = GGUFModel(path)
            atexit.register(self.model.close)  # llama.cpp aborts at exit otherwise (D40)
        return self.model

    def submit(self, text: str, url: str) -> dict:
        cfg = load_config("data")["chunking"]
        n = len(chunk_text(clean_text(text), cfg["target_tokens"], cfg["overlap_tokens"]))
        with self.lock:
            job_id = str(next(self.ids))
            self.jobs[job_id] = {"status": "queued", "done": 0, "total": n, "url": url, "text": text,
                                 "created": time.time()}
            for old in sorted(self.jobs, key=lambda k: self.jobs[k]["created"])[:-KEEP_JOBS]:
                if self.jobs[old]["status"] in ("done", "error", "cancelled"):
                    del self.jobs[old]
        self.queue.put(job_id)
        return {"job": job_id, "n_chunks": n}

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job["status"] not in ("queued", "running"):
            return False
        job["cancel"] = True
        if job["status"] == "queued":
            job["status"] = "cancelled"
        return True

    def view(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return {k: v for k, v in job.items() if k in ("status", "done", "total", "result", "error", "url")}

    def _run(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job["status"] == "cancelled":
                continue
            job["status"] = "running"
            try:
                model = self._load()

                def predict(chunk: str, job: dict = job, model=model) -> dict:
                    if job.get("cancel"):
                        raise Cancelled
                    r = model.analyse(chunk)
                    job["done"] += 1
                    return r

                r = analyse_document(job.pop("text"), predict)
                keep = ("category", "severity", "explanation", "quote", "quote_found")
                job["result"] = {**{k: r[k] for k in ("grade", "score", "counts", "n_chunks", "chars", "seconds",
                                                      "note")},
                                 "flags": [{k: f.get(k) for k in keep} for f in r["flags"]]}
                job["status"] = "done"
            except Cancelled:
                job["status"] = "cancelled"
            except Exception as e:  # noqa: BLE001 -- report any failure to the extension, keep the worker alive
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
            finally:
                job.pop("text", None)


def make_handler(worker: Worker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _allowed(self) -> bool:
            origin = self.headers.get("Origin")
            return origin is None or origin.startswith(EXTENSION_ORIGINS)

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            origin = self.headers.get("Origin")
            if origin and origin.startswith(EXTENSION_ORIGINS):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self) -> bool:
            if not self._allowed():
                self._send(403, {"error": "only the Fine Print extension may use this server"})
                return False
            return True

        def do_OPTIONS(self) -> None:
            if self._guard():
                self._send(204, {})

        def do_GET(self) -> None:
            if not self._guard():
                return
            if self.path == "/health":
                self._send(200, {"ok": True, "model": os.path.basename(worker.model_path or MODEL_FILE),
                                 "loaded": worker.model is not None, "queue": worker.queue.qsize(),
                                 "max_chars": MAX_CHARS})
            elif self.path.startswith("/jobs/") and (job := worker.view(self.path[6:])) is not None:
                self._send(200, job)
            else:
                self._send(404, {"error": "not found"})

        def do_DELETE(self) -> None:
            if self._guard():
                ok = self.path.startswith("/jobs/") and worker.cancel(self.path[6:])
                self._send(200 if ok else 404, {"cancelled": ok})

        def do_POST(self) -> None:
            if not self._guard():
                return
            if self.path != "/analyse":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 4 * MAX_CHARS:
                self._send(413, {"error": f"page too long: up to {MAX_CHARS:,} characters"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "body must be JSON"})
                return
            text = (body.get("text") or "").strip()
            if not text:
                self._send(400, {"error": "no text"})
            elif len(text) > MAX_CHARS:
                self._send(413, {"error": f"{len(text):,} characters; up to {MAX_CHARS:,}. Select one section."})
            else:
                self._send(202, worker.submit(text, str(body.get("url") or "")))

        def log_message(self, fmt: str, *args) -> None:
            if not self.path.startswith("/jobs/"):  # polling would flood the terminal
                super().log_message(fmt, *args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="GGUF file (default: $FINEPRINT_MODEL, else download from Hugging Face)")
    ap.add_argument("--port", type=int, default=8737)
    args = ap.parse_args()
    worker = Worker(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(worker))
    print(f"Fine Print server on http://127.0.0.1:{args.port} (model loads on the first request; Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
