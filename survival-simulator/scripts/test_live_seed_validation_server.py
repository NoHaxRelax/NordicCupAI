"""Short integration check; full-domain performance is measured separately."""
import os
os.environ["SEED_SEARCH_AT"] = "99999"
os.environ["SEED_SHADOW_RUN_DIR"] = "/tmp/live-seed-validation-test"
import pathlib
import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastsim import SimulationCore
from scripts.live_seed_validation_server import LiveGame

sim = SimulationCore(seed=12345)
live = LiveGame()
state = {"game_status":"ok", "score":0, "sim_time":0, "n_agents":5, "agent_status":[]}
for _ in range(50):
    response = live.predict(state)
    state = sim.step([(a["agent_id"], a) for a in response["actions"]])
    state["agent_status"] = state.pop("observations")
assert live.status()["frames"] == 49
assert live.status()["policy"] == "expanded_local_food"
assert live.status()["recovery"] == "gathering"
assert live.status()["latency_max_ms"] < 1000
print(live.status())
