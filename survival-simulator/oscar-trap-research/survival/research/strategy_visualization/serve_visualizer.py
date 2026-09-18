"""Serve the strategy viewer and a dedicated read-only Codex Sol helper."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import importlib.util
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
SURVIVAL = HERE.parents[1]
THREAD_FILE = Path("/tmp/predator-strategy-sol-helper-thread")
STOP_FILE = Path("/tmp/predator-intake-stop")
MODEL = "gpt-5.6-sol"
QUESTION_LIMIT = 3000
LOCK = threading.Lock()
USAGE_PATH = SURVIVAL / "research/intake_validation/usage.py"

START_PROMPT = """You are the dedicated Sol research helper embedded in a local predator-strategy visualizer.
You are a separate helper session: do not claim to be the main agent or any simulation worker.
Work read-only. Never edit files, launch simulations, start background jobs, send messages, access credentials,
or use the network. Answer questions using evidence already present under this Survival research directory.
Prioritize research/simple_chase/SESSION3_RESULTS.md, research/reliability_eval/README.md,
results/reliability_eval/development_scores, research/backup_guides, research/replaceable_sites,
research/reacquisition_sol, and the latest original receipts. Older context lives in
research/simple_chase/SESSION2_RESULTS.md, ../docs/predator-sequential-intake.md and
../docs/survival-wall-funneling.md. Do not pool different policy versions, fitted repairs,
fresh tests, prepared crowd retention, and random-map transport. Clearly distinguish arranged fixtures, native pixels,
offline upstream-renderer reconstructions, pilot horizons, full 3,000-second games, observed policy inputs,
and evaluator-only hidden state. Treat /tmp/predator-intake-stop as a hard stop for any response.
Keep answers concise, candid, and useful. Cite repository-relative files when a claim may need inspection.

Current visualizer selection: {strategy}
Question: {question}
"""


def parse_thread_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            return event.get("thread_id")
    return None


def quota_stop_message() -> str | None:
    threshold = 60
    try:
        spec = importlib.util.spec_from_file_location("visualizer_usage", USAGE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        threshold = getattr(module, "STOP_REMAINING", threshold)
        if STOP_FILE.exists():
            return f"The shared {threshold}% quota stop marker is active, so this helper will not start a new model call."
        status = module.status()
        remaining = status.get("remaining_percent", {})
        threshold = status.get("stop_remaining", threshold)
        if status.get("stop") or (remaining and min(remaining.values()) <= threshold):
            return f"The {threshold}% account quota threshold has been reached ({remaining}); this helper will not start a new model call."
    except Exception:
        if STOP_FILE.exists():
            return f"The shared {threshold}% quota stop marker is active, so this helper will not start a new model call."
    return None


def ask_sol(question: str, strategy: str) -> tuple[str, bool]:
    if message := quota_stop_message():
        return message, False
    codex = os.environ.get("CODEX_BIN", "codex")
    thread_id = THREAD_FILE.read_text().strip() if THREAD_FILE.exists() else ""
    with tempfile.NamedTemporaryFile(prefix="sol-helper-", suffix=".txt", delete=False) as output:
        output_path = Path(output.name)
    try:
        if thread_id:
            prompt = (f"Current visualizer selection: {strategy}\nQuestion: {question}\n"
                      "Remember: read-only, concise, and evidence-backed. Check research/simple_chase/SESSION3_RESULTS.md "
                      "and current receipt/audit files for new progress; older conversation results may be stale. "
                      "Keep policy versions, fitted versus fresh tests, and prepared holding versus transport separate.")
            command = [codex, "-a", "never", "-s", "read-only", "exec", "resume", "-m", MODEL,
                       "--json", "-o", str(output_path), thread_id, "-"]
        else:
            prompt = START_PROMPT.format(strategy=strategy, question=question)
            command = [codex, "-a", "never", "-s", "read-only", "exec", "-m", MODEL, "-C", str(SURVIVAL),
                       "--json", "-o", str(output_path), "-"]
        process = subprocess.run(command, input=prompt, text=True, capture_output=True, timeout=180)
        answer = output_path.read_text(errors="replace").strip()
        if process.returncode != 0:
            detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "Codex exited without an answer."
            return f"The dedicated Sol helper could not answer: {detail}", False
        if not thread_id:
            thread_id = parse_thread_id(process.stdout) or ""
            if thread_id:
                THREAD_FILE.write_text(thread_id + "\n")
        if not answer:
            return "The dedicated Sol helper returned an empty response.", False
        return answer, True
    except subprocess.TimeoutExpired:
        return "The dedicated Sol helper timed out after three minutes. Try a narrower question.", False
    finally:
        output_path.unlink(missing_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SURVIVAL), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_error(404)
            return
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self.send_json(415, {"ok": False, "answer": "Content-Type must be application/json."})
            return
        port = self.server.server_address[1]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        if host not in allowed_hosts:
            self.send_json(403, {"ok": False, "answer": "Host is not allowed."})
            return
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != "http" or parsed.netloc not in allowed_hosts:
                self.send_json(403, {"ok": False, "answer": "Cross-origin requests are not allowed."})
                return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 12_000:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            question = str(payload.get("question", "")).strip()
            strategy = str(payload.get("strategy", "No strategy selected")).strip()[:300]
            if not question or len(question) > QUESTION_LIMIT:
                raise ValueError(f"Questions must contain 1–{QUESTION_LIMIT} characters")
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"ok": False, "answer": str(error)})
            return
        with LOCK:
            answer, ok = ask_sol(question, strategy)
        self.send_json(200 if ok else 503, {"ok": ok, "answer": answer,
                                          "helper": "Dedicated gpt-5.6-sol session · read-only"})

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9055)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Strategy visualizer + Sol helper: http://127.0.0.1:{args.port}/research/strategy_visualization/index.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
