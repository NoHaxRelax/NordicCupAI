import json
import logging
import time
from pathlib import Path
from threading import Lock

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

import os
from night_policy import NightPolicy
from harvest import Harvester
CONFIG = json.loads(Path(os.environ.get("NIGHT_CONFIG", str(Path(__file__).with_name("st_any.json")))).read_text())
app = FastAPI(title="nightsim C++ policy endpoint")
def _parse_harvest(raw):
    """systemd strips double quotes from Environment= values, so accept {budget:20000,max_harvests:16} too."""
    raw = (raw or "").strip()
    if not raw or raw == "{}":
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        out = {}
        for part in raw.strip("{} ").split(","):
            key, _, value = part.partition(":")
            key = key.strip().strip("'\"")
            if key:
                out[key] = int(float(value.strip().strip("'\"")))
        return out


HARVEST = _parse_harvest(os.environ.get("NIGHT_HARVEST"))   # empty = off
policy = NightPolicy(CONFIG)
harvester = Harvester(enabled=bool(HARVEST), **HARVEST)
last_time = None
lock = Lock()
run_started_ns = None
handler_ms = []
policy_ms = []
agent_counts = []
observation_counts = []
LATENCY_LOG = Path("/var/log/nordiccup-validation-latency.jsonl")
log = logging.getLogger("nordiccup.latency")


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * pct)]


def summarize(values):
    return {
        "min_ms": min(values, default=0.0),
        "mean_ms": sum(values) / len(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values, default=0.0),
    }


def finish_run(now_ns):
    global run_started_ns
    record = {
        "finished_unix": time.time(),
        "requests": len(handler_ms),
        "wall_seconds": (now_ns - run_started_ns) / 1e9 if run_started_ns else 0.0,
        "handler": summarize(handler_ms),
        "native_policy": summarize(policy_ms),
        "agents": summarize(agent_counts),
        "observations": summarize(observation_counts),
    }
    LATENCY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LATENCY_LOG.open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.warning("VALIDATION_LATENCY %s", json.dumps(record, separators=(",", ":")))
    run_started_ns = None
    handler_ms.clear(); policy_ms.clear(); agent_counts.clear(); observation_counts.clear()
    return record


@app.post("/predict")
def predict(step: dict = Body(...)):
    global policy, last_time, run_started_ns
    request_started = time.perf_counter_ns()
    if run_started_ns is None:
        run_started_ns = request_started
    sim_time = float(step.get("sim_time", 0.0))
    game_status = str(step.get("game_status", "running"))
    with lock:
        if last_time is not None and sim_time < last_time:
            policy = NightPolicy(CONFIG); harvester.reset()
        # Keep the HTTP boundary deliberately permissive. The C++ adapter validates
        # and consumes only the public fields used by the policy; extra envelope
        # fields from the competition API cannot turn a reachable request into 422.
        states = list(step.get("agent_status", []))
        native_started = time.perf_counter_ns()
        actions = policy(states, sim_time) if states else []
        if states and harvester.enabled:
            actions = harvester.apply(states, sim_time, actions)
        native_elapsed = (time.perf_counter_ns() - native_started) / 1e6
        last_time = sim_time
        policy_ms.append(native_elapsed)
        agent_counts.append(len(states))
        observation_counts.append(sum(len(state.get("observations", [])) for state in states))
        elapsed = (time.perf_counter_ns() - request_started) / 1e6
        handler_ms.append(elapsed)
        summary = None
        if game_status == "game_over":
            policy, last_time = NightPolicy(CONFIG), None; harvester.reset()
            summary = finish_run(time.perf_counter_ns())
    headers = {
        "Server-Timing": f'policy;dur={native_elapsed:.3f}, handler;dur={elapsed:.3f}',
        "X-NordicCup-Handler-Ms": f"{elapsed:.3f}",
    }
    if summary is not None:
        headers["X-NordicCup-Run-Requests"] = str(summary["requests"])
    return JSONResponse({"actions": actions}, headers=headers)


@app.get("/")
def index():
    return {"policy": "nightsim", "config": os.environ.get("NIGHT_CONFIG", "st_any.json"), "harvest": HARVEST, "harvests_done": harvester.harvests, "runtime": "native-cpp"}
