"""Run the HTTP policy/harvest logic against the published Python engine locally."""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/bumblebee/Github_repos/Projects/NordicCupAI/survival-simulator")

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
import nightsim
from nightsim.serve.harvest import Harvester
from nightsim.serve.night_policy import NightPolicy


def run(cfg, seed, budget, max_harvests, selector, trigger, closing, horizon):
    sim = SimulationCore(seed=seed)
    state = sim.step([])
    policy = NightPolicy(cfg)
    hv = Harvester(enabled=True, budget=budget, max_harvests=max_harvests,
                   sacrifice_mode=selector, trigger=trigger, closing=closing)
    large_bursts = 0
    successful = 0
    min_pred_energy = 0.0
    while state["observations"] and state["sim_time"] < horizon:
        base = policy(state["observations"], state["sim_time"])
        out = hv.apply(state["observations"], state["sim_time"], base)
        if len(out) >= budget:
            large_bursts += 1
        state = sim.step([(a["agent_id"], ActionRequest(**a)) for a in out])
        for p in sim.env.predators:
            min_pred_energy = min(min_pred_energy, p.energy)
        successful = sum(1 for p in sim.env.predators if p.energy < -1000)
    return {
        "seed": seed, "score": round(sim.env.score, 3),
        "sim_time": round(sim.env.time, 1), "harvests": hv.harvests,
        "large_bursts": large_bursts, "successful_negative_predators": successful,
        "min_pred_energy": round(min_pred_energy, 1), "hlog": hv.log,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seeds", required=True)
    ap.add_argument("--budget", type=int, required=True); ap.add_argument("--max-harvests", type=int, default=3)
    ap.add_argument("--selector", default="low_energy"); ap.add_argument("--trigger", type=float, default=30.)
    ap.add_argument("--closing", type=float, default=6.)
    ap.add_argument("--horizon", type=float, default=300.)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    seeds = [int(x) for x in args.seeds.split(",")]
    for seed in seeds:
        print(json.dumps(run(cfg, seed, args.budget, args.max_harvests,
                             args.selector, args.trigger, args.closing, args.horizon)), flush=True)
