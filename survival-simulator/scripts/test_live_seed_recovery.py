"""End-to-end bounded-domain analogue of the live validation flow."""
import os
os.environ["SEED_SEARCH_AT"] = "30"
os.environ["SEED_MIN_SAMPLES"] = "50"
os.environ["SEED_SEARCH_END"] = "65536"
os.environ["SEED_SEARCH_WORKERS"] = "4"
os.environ["SEED_WAIT_PER_ACTION"] = "9"
os.environ["SEED_SHADOW_RUN_DIR"] = "/tmp/live-seed-recovery-test"
import pathlib
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastsim import SimulationCore
from scripts.live_seed_validation_server import LiveGame

sim = SimulationCore(seed=12345)
live = LiveGame()
state = {"game_status":"ok", "score":0, "sim_time":0, "n_agents":5, "agent_status":[]}
for _ in range(800):
    response = live.predict(state)
    state = sim.step([(a["agent_id"], a) for a in response["actions"]])
    state["agent_status"] = state.pop("observations")
    if live.recovery in ("recovered", "failed", "desynchronized"):
        break
if live.search_thread:
    live.search_thread.join(timeout=10)
status = live.status()
assert status["recovery"] == "recovered", status
assert status["recovered_seed"] == 12345, status
assert status["shadow_status"] == "public_consistent", status
assert status["latency_max_ms"] < 10000, status
print(status)
