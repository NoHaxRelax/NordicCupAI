"""Local native-engine sweep for the score-guarded duplicate-action layer."""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent / "nightsim" / "serve"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import nightsim
from harvest import Harvester


def one(job):
    cfg, seed, horizon, hv_kw, score_guard, score_reserve, safe_mode = job
    started = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=True)
    eng = sim._engine
    state = sim.step([])
    eng.policy_init(nightsim.seed_key(seed), dict(cfg))
    hv = Harvester(**hv_kw)
    guard_tripped = False
    bursts = 0
    transfers = 0
    burst_scores = []
    while state["observations"] and state["sim_time"] < horizon:
        base = [{"agent_id": a, "move_distance": d, "move_direction": di,
                 "turn_angle": t, "spawn_agent": sp}
                for a, d, di, t, sp in eng.policy_act()]
        score_before = float(eng.info()["score"])
        if score_guard > 0 and score_before + score_reserve >= score_guard:
            guard_tripped = True
        if guard_tripped and safe_mode:
            out = [{"agent_id": a["agent_id"], "move_distance": 0.0,
                    "move_direction": 0.0, "turn_angle": 0.0,
                    "spawn_agent": False} for a in base]
        else:
            out = base if guard_tripped else hv.apply(state["observations"], state["sim_time"], base)
        burst_index = None
        if len(hv.log) > bursts:
            bursts += 1
            burst_scores.append({"before": round(score_before, 3), "actions": len(out)})
            burst_index = bursts - 1
        state = sim.step([(a["agent_id"], a) for a in out])
        event_energies = []
        for kind, _t, _aid, _age, energy in eng.pop_events():
            if kind == "predator":
                event_energies.append(round(energy, 3))
            if kind == "predator" and energy < -1000:
                transfers += 1
        if burst_index is not None:
            burst_scores[burst_index].update(after=round(float(eng.info()["score"]), 3),
                                             predator_energies=event_energies)
    info = eng.info()
    return {
        "seed": seed,
        "score": round(info["score"], 3),
        "surv": round(info["time"], 1),
        "harvests": hv.harvests,
        "bursts": bursts,
        "transfers": transfers,
        "guard_tripped": guard_tripped,
        "burst_scores": burst_scores,
        "wall": round(time.perf_counter() - started, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--max-harvests", type=int, default=12)
    ap.add_argument("--cooldown", type=float, default=50.0)
    ap.add_argument("--selector", default="low_energy")
    ap.add_argument("--trigger", type=float, default=30.0)
    ap.add_argument("--closing", type=float, default=6.0)
    ap.add_argument("--horizon", type=float, default=3000.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--score-guard", type=float, default=2300.0)
    ap.add_argument("--score-reserve", type=float, required=True)
    ap.add_argument("--safe-mode", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    hv_kw = dict(enabled=True, budget=args.budget, max_harvests=args.max_harvests,
                 sacrifice_mode=args.selector, trigger=args.trigger,
                 closing=args.closing, cooldown=args.cooldown)
    jobs = [(cfg, seed, args.horizon, hv_kw, args.score_guard, args.score_reserve, args.safe_mode)
            for seed in seeds]
    started = time.time()
    with mp.Pool(args.workers) as pool, open(args.out, "w") as out:
        for row in pool.imap_unordered(one, jobs):
            out.write(json.dumps(row) + "\n")
            out.flush()
    print(f"done {len(jobs)} runs in {time.time() - started:.0f}s -> {args.out}", flush=True)
