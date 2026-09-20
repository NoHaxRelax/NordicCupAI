#!/usr/bin/env python3
"""Run an unmodified Python game against a live /predict service."""
import argparse, json, time
from pathlib import Path

import requests

from src.core import SimulationCore
from src.utils.DTOs import StepResponse, ObservationResponse, ActionRequest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--timeout", type=float, default=9.85)
    args = p.parse_args()
    sim = SimulationCore(seed=args.seed)
    state = StepResponse(game_status="ok", score=0, sim_time=sim.env.time,
                         n_agents=len(sim.env.agents), agent_status=[])
    started = time.monotonic(); ticks = 0; max_latency = 0.; wait_total = 0.
    while state.game_status != "game_over":
        before = time.monotonic()
        response = requests.post(args.url, json=state.dict(), timeout=args.timeout)
        latency = time.monotonic() - before
        max_latency = max(max_latency, latency); wait_total += latency
        response.raise_for_status()
        actions = [ActionRequest(**a) for a in response.json().get("actions", [])]
        raw = sim.step([(a.agent_id, a) for a in actions])
        status = [ObservationResponse(
            agent_id=o["agent_id"], observations=o["observations"], energy=o["energy"],
            biome=o["biome"], age=o["age"], speed=o["speed"],
            sprint_speed=o["sprint_speed"], hearing_radius=o["hearing_radius"],
            vision_angle=o["vision_angle"], vision_range=o["vision_range"],
            max_energy=o["max_energy"]) for o in raw["observations"] if o is not None]
        state = StepResponse(game_status="ok" if status and sim.env.time <= 3000 else "game_over",
                             score=raw["score"], sim_time=raw["sim_time"],
                             n_agents=raw["num_agents"], agent_status=status)
        ticks += 1
        if ticks % 1000 == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"seed": args.seed, "time": sim.env.time,
                "score": state.score, "agents": len(status), "ticks": ticks,
                "wall_seconds": time.monotonic()-started, "wait_seconds": wait_total}))
    result = {"seed": args.seed, "score": state.score, "duration": sim.env.time,
              "ticks": ticks, "wall_seconds": time.monotonic()-started,
              "request_wait_seconds": wait_total, "max_request_seconds": max_latency}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
