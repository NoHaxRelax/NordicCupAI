"""Serving endpoint, shaped like the organiser's 2025 race-car contract.

Their evaluator POSTs the observation to /predict and expects actions back. The
request body is the payload we captured, so it is parsed leniently (extra fields
are kept, missing ones default) and handed to the trained policy. The response
DTO is a best guess at their shape and is the ONE thing expected to change on
day one, so it is isolated in `make_response`.

    uvicorn serve.api:app --host 0.0.0.0 --port 9052

Env vars:
    CKPT        path to a checkpoint (default runs/*/best.pt, newest)
    GREEDY      1 = argmax, 0 = sample (default 1)
    PLAN_K      how many actions to return per agent per call (race-car asked
                for a batch to cut latency; 1 = one action per tick)
"""

from __future__ import annotations

import datetime
import glob
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survivalsim.encoder import EntityPolicy, featurize_agent  # noqa: E402
from survivalsim.env import ACTIONS  # noqa: E402

HOST, PORT = "0.0.0.0", 9052
GREEDY = os.environ.get("GREEDY", "1") == "1"
PLAN_K = int(os.environ.get("PLAN_K", "1"))


# ------------------------------------------------------------------- DTOs
# Lenient on purpose: `extra="allow"` means any field the real payload adds on
# the day is kept and can be forwarded to the featurizer, not rejected.


class AgentStatus(BaseModel):
    model_config = ConfigDict(extra="allow")
    agent_id: int
    observations: list[dict[str, Any]] = []
    energy: float = 0.0
    max_energy: float = 1.0
    biome: str = ""
    age: float = 0.0
    speed: float = 1.0
    sprint_speed: float = 1.0
    hearing_radius: float = 0.0
    vision_angle: float = 0.0
    vision_range: float = 0.0


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    game_status: str = "running"
    score: float = 0.0
    agent_status: list[AgentStatus] = []


class AgentAction(BaseModel):
    agent_id: int
    actions: list[str]


class PredictResponse(BaseModel):
    actions: list[AgentAction]


# ----------------------------------------------------------------- policy


def _find_ckpt() -> str:
    if os.environ.get("CKPT"):
        return os.environ["CKPT"]
    cands = sorted(glob.glob("runs/*/best.pt"), key=os.path.getmtime)
    if not cands:
        raise SystemExit("no checkpoint: set CKPT or train a run first")
    return cands[-1]


CKPT = _find_ckpt()
_ck = torch.load(CKPT, map_location="cpu", weights_only=False)
_cfg = _ck.get("config", {})
model = EntityPolicy(d_model=_cfg.get("d_model", 64), n_layers=_cfg.get("n_layers", 2))
model.load_state_dict(_ck["model"])
model.eval()
# Requests carry a handful of agents with under ten entities each; measured
# thread spin-up dominates above ~4 threads on batches this small.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


@torch.no_grad()
def decide(agents: list[dict]) -> list[list[str]]:
    """One forward pass for every agent in the request; returns PLAN_K actions each.

    With PLAN_K > 1 the same action is repeated, which is the honest thing a
    reactive policy can do without a model of what happens between calls. If
    their evaluator wants a real multi-step plan, this is where an imagined
    rollout would slot in.
    """
    if not agents:
        return []
    b = {k: torch.from_numpy(np.stack(v)) for k, v in zip(
        ("types", "feats", "mask", "body"), zip(*(featurize_agent(a) for a in agents)))}
    logits, _ = model(**b)
    idx = logits.argmax(-1) if GREEDY else torch.distributions.Categorical(logits=logits).sample()
    return [[ACTIONS[int(i)]] * PLAN_K for i in idx]


def make_response(req: PredictRequest, plans: list[list[str]]) -> PredictResponse:
    """The one function to rewrite when we see their real response DTO."""
    return PredictResponse(actions=[
        AgentAction(agent_id=a.agent_id, actions=p) for a, p in zip(req.agent_status, plans)
    ])


# -------------------------------------------------------------------- app

app = FastAPI()
start_time = time.time()
_calls = 0
_lat_ms = 0.0


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest = Body(...)) -> PredictResponse:
    global _calls, _lat_ms
    t0 = time.perf_counter()
    agents = [a.model_dump() for a in request.agent_status]
    plans = decide(agents) if request.game_status == "running" else [["NOTHING"] * PLAN_K for _ in agents]
    _calls += 1
    _lat_ms += (time.perf_counter() - t0) * 1000
    return make_response(request, plans)


@app.get("/api")
def info() -> dict:
    return {
        "service": "survival-usecase",
        "uptime": str(datetime.timedelta(seconds=int(time.time() - start_time))),
        "checkpoint": CKPT,
        "calls": _calls,
        "mean_latency_ms": round(_lat_ms / _calls, 2) if _calls else None,
        "greedy": GREEDY, "plan_k": PLAN_K,
    }


@app.get("/")
def index() -> str:
    return "Your endpoint is running!"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve.api:app", host=HOST, port=PORT)
