"""FastAPI endpoint matching the 2025 competition template.

FastAPI and pydantic are imported lazily so that the entire core of this
package -- schema, belief, policies, mock -- stays importable and testable
with the standard library alone.

The handler is intentionally thin. Everything it does is: parse tolerantly,
update belief, ask the policy, encode, capture. If the wire format turns out
to differ, the change lands in `actions.py`; if the schema differs, in
`schema.py`. Neither should require touching this file.
"""

from __future__ import annotations

import datetime
import os
import time
from dataclasses import asdict
from typing import Any

from actions import get_codec
from capture import Capture
from schema import GameState

HOST = os.environ.get("SP_HOST", "0.0.0.0")
PORT = int(os.environ.get("SP_PORT", "9060"))
POLICY_NAME = os.environ.get("SP_POLICY", "belief")
CODEC_NAME = os.environ.get("SP_CODEC", "continuous")
CAPTURE_PATH = os.environ.get("SP_CAPTURE", "runs/live.jsonl")


def create_app():
    from fastapi import Body, FastAPI

    import policies

    app = FastAPI(title="speculative_preparedness")
    start = time.time()
    policy = policies.build(POLICY_NAME)
    codec = get_codec(CODEC_NAME)
    capture = Capture(CAPTURE_PATH, policy=POLICY_NAME, codec=codec.name)
    last_seen = {"t": time.time()}

    @app.post("/predict")
    def predict(payload: dict[str, Any] = Body(...)) -> Any:
        now = time.time()
        dt = min(max(now - last_seen["t"], 1e-3), 5.0)
        last_seen["t"] = now

        state = GameState.parse(payload)
        actions = policy.act(state, dt=dt)
        wire = codec.encode(actions)

        belief_snapshot = []
        if hasattr(policy, "beliefs"):
            belief_snapshot = [b.snapshot() for b in policy.beliefs]

        capture.record(
            payload, state=state, actions=actions, wire=wire, belief=belief_snapshot
        )
        return wire

    @app.get("/api")
    def meta() -> dict[str, Any]:
        return {
            "service": "speculative-preparedness",
            "policy": POLICY_NAME,
            "codec": codec.name,
            "capture": CAPTURE_PATH,
            "uptime": str(datetime.timedelta(seconds=time.time() - start)),
        }

    @app.get("/")
    def index() -> str:
        return "Your endpoint is running!"

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host=HOST, port=PORT)
