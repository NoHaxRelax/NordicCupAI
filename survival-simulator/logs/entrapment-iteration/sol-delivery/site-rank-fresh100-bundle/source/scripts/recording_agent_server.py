"""
Agent server that answers with the dummy policy and records every tick it receives.

Each request body is stored byte for byte, together with the headers, arrival time,
response time and the actions sent back. Open http://localhost:9052/viewer to browse
the ticks (it follows the latest tick live while a game runs).

Usage (from the survival-simulator folder):
    python scripts/recording_agent_server.py [--port 9052] [--log-dir logs/ticks]

The competition server can POST to either "/" or "/predict".
Every game gets its own JSONL file. A new game starts when sim_time goes backwards or
after NEW_GAME_GAP seconds without requests. The body is recorded before any parsing, so
payloads that don't match src/utils/DTOs.py are still logged (the live server omits
sim_time and n_agents, for example).
"""
import argparse
import json
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.controllers.dummy_agent_policy import action_decision  # noqa: E402

VIEWER_HTML = Path(__file__).with_name("tick_viewer.html")
NEW_GAME_GAP = 30.0


class TickRecorder:
    """Appends ticks to one JSONL file per game and keeps byte offsets for random access."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.runs = {}  # run name -> list of byte offsets
        self.current = None
        self.last_sim_time = None
        self.last_received = 0.0
        for path in sorted(self.log_dir.glob("*.jsonl")):
            self.runs[path.stem] = self._index(path)

    @staticmethod
    def _index(path: Path):
        offsets, pos = [], 0
        with path.open("rb") as f:
            for line in f:
                offsets.append(pos)
                pos += len(line)
        return offsets

    def record(self, entry: dict, sim_time):
        with self.lock:
            went_back = sim_time is not None and self.last_sim_time is not None and sim_time < self.last_sim_time
            idle = entry["received_at"] - self.last_received > NEW_GAME_GAP
            if self.current is None or went_back or idle:
                self.current = datetime.now().strftime("game_%Y%m%d_%H%M%S_%f")
                self.runs[self.current] = []
            self.last_sim_time = sim_time
            self.last_received = entry["received_at"]
            path = self.log_dir / f"{self.current}.jsonl"
            entry["tick"] = len(self.runs[self.current])
            line = (json.dumps(entry) + "\n").encode()
            with path.open("ab") as f:
                self.runs[self.current].append(f.tell())
                f.write(line)

    def read(self, run: str, tick: int):
        with self.lock:
            offsets = self.runs.get(run)
            if offsets is None or not 0 <= tick < len(offsets):
                raise HTTPException(404, "tick not found")
            offset = offsets[tick]
        with (self.log_dir / f"{run}.jsonl").open("rb") as f:
            f.seek(offset)
            return json.loads(f.readline())


def create_app(recorder: TickRecorder) -> FastAPI:
    app = FastAPI(title="Survival Simulator Recording Agent Endpoint")

    async def predict(request: Request):
        received = time.time()
        raw = await request.body()
        entry = {
            "received_at": received,
            "client": request.client.host if request.client else None,
            "path": request.url.path,
            "headers": dict(request.headers),
            "body_bytes": len(raw),
        }
        body = response = None
        try:
            body = json.loads(raw)
            rng = random.Random(1)  # same as agent_server.py
            actions = [action_decision(agent, rng).model_dump() for agent in body.get("agent_status") or []]
            response = {"actions": actions}
            return response
        except Exception as e:
            entry["error"] = repr(e)
            raise
        finally:
            entry["handled_ms"] = (time.time() - received) * 1000
            entry["request"] = body if body is not None else raw.decode(errors="replace")
            entry["response"] = response
            recorder.record(entry, body.get("sim_time") if isinstance(body, dict) else None)

    app.add_api_route("/predict", predict, methods=["POST"])
    app.add_api_route("/", predict, methods=["POST"])

    @app.get("/")
    def index():
        return {"message": "Agent endpoint running!"}

    @app.get("/viewer")
    def viewer():
        return FileResponse(VIEWER_HTML)

    @app.get("/api/runs")
    def runs():
        with recorder.lock:
            return [{"name": name, "ticks": len(offsets)} for name, offsets in sorted(recorder.runs.items())]

    @app.get("/api/runs/{run}/ticks/{tick}")
    def tick(run: str, tick: int):
        return recorder.read(run, tick)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9052)
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parents[1] / "logs" / "ticks")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(create_app(TickRecorder(args.log_dir)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
