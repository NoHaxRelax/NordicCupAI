#!/usr/bin/env python3
"""Record every native-policy tick for the existing entrapment sprite viewer."""

import argparse
import gzip
import hashlib
import json
import math
import os
import pathlib
import platform
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = pathlib.Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_policy"
sys.path.insert(0, str(NATIVE))
sys.path.insert(0, str(ROOT))

import nightsim


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges_from(obstacles):
    edges = []
    for x, y, width, height in obstacles:
        corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
        edges.extend([[list(corners[i]), list(corners[(i + 1) % 4])] for i in range(4)])
    return edges


def load_config(path, name):
    configs = json.loads(path.read_text())
    if name is None:
        if len(configs) != 1:
            raise ValueError("config has multiple variants; pass --config-name")
        return next(iter(configs.items()))
    return name, configs[name]


def world(engine):
    agents = []
    for aid, x, y, direction, age, energy, max_energy, speed, sprint, hearing, vision, _, cone, _ in engine.agents():
        agents.append(dict(agent_id=aid, x=x, y=y, direction=direction, size=5,
                           color=[128, 128, 128], energy=energy, max_energy=max_energy,
                           age=age, speed=speed, sprint_speed=sprint,
                           hearing_radius=hearing, vision_radius=vision,
                           cone_angle=cone, _vision_poly=None))
    predators = [dict(x=x, y=y, direction=direction, size=10, color=[255, 0, 0],
                      energy=energy, max_energy=200.0, age=0.0, hearing_radius=60.0,
                      vision_radius=250.0, cone_angle=math.pi / 3, _vision_poly=None)
                 for x, y, direction, energy, _ in engine.predators()]
    fruits = [dict(x=x, y=y, radius=radius, color=[0, 255, 0], age=age)
              for _, x, y, _, age, radius, _ in engine.fruits()]
    trees = [dict(x=x, y=y, radius=radius, color=[82, 22, 12], age=age)
             for x, y, radius, age in engine.trees()]
    return dict(agents=agents, predators=predators, fruits=fruits, trees=trees)


def policy_overlay(engine):
    agents = {row[0] for row in engine.agents()}
    roles = {str(aid): "gatherer" for aid in agents}
    sites = []
    guides = {}
    bait = incoming = None
    for row in engine.dbg_roles():
        _, has_trap, group_bait, replacement, guide, guide_state, *_ = row
        if group_bait in agents:
            roles[str(group_bait)] = "bait"
            bait = group_bait if bait is None else bait
        if replacement in agents:
            roles[str(replacement)] = "replacement_bait"
            incoming = replacement if incoming is None else incoming
        if guide in agents:
            roles[str(guide)] = "guide"
            guides[str(guide)] = {"state": guide_state}
        if has_trap:
            sites.append(row[0])
    trap = engine.dbg_trap()
    site = None
    if trap is not None:
        site = {"mouth": [trap[0], trap[1]], "goal": [trap[2], trap[3]]}
    phase = "entrapment" if site else "exploration"
    return dict(phase=phase, site=site, site_group=sites[0] if sites else None,
                bait=bait, incoming=incoming, roles=roles, metrics={}, map={}, guides=guides)


def current_near_bait(engine, overlay):
    bait_ids = {value for value in (overlay["bait"], overlay["incoming"]) if value is not None}
    bait_positions = [(row[1], row[2]) for row in engine.agents() if row[0] in bait_ids]
    if not bait_positions:
        return 0
    return sum(any(math.hypot(px - bx, py - by) <= 40 for bx, by in bait_positions)
               for px, py, *_ in engine.predators())


def write_background(seed, path):
    """Render only static terrain using the deterministic reference renderer."""
    import pygame
    from src.core import SimulationCore as PythonSimulationCore
    simulation = PythonSimulationCore(seed=seed)
    background = simulation.env.static_surface.copy()
    background.blit(simulation.env.shadow_surface, (0, 0))
    background.blit(simulation.env.obstacle_surface, (0, 0))
    pygame.image.save(background, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--config-name")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seconds", type=float, default=3000.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if (args.out / "summary.json").exists():
        parser.error("output already contains a completed replay")

    config_path = args.config.resolve()
    config_name, params = load_config(config_path, args.config_name)
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = args.out / "chunks"
    chunks.mkdir(exist_ok=True)
    write_background(args.seed, args.out / "background.png")

    sim = nightsim.SimulationCore(seed=args.seed, predators=True)
    engine = sim._engine
    sim.step([])
    engine.policy_init(nightsim.seed_key(args.seed), params)
    obstacles = engine.obstacles()
    atomic_json(args.out / "static.json", dict(width=1600, height=1200,
                                                edges=edges_from(obstacles)))
    sources = [pathlib.Path(__file__).resolve(), config_path, NATIVE / "evaluate.py",
               NATIVE / "nightsim" / "__init__.py"] + sorted(
                   path for path in (NATIVE / "nightsim").iterdir()
                   if path.suffix in {".cpp", ".h", ".hpp"})
    atomic_json(args.out / "manifest.json", {
        "seed": args.seed, "horizon": args.seconds, "dt": sim.dt,
        "config": {"path": str(config_path), "name": config_name,
                   "sha256": sha256(config_path)},
        "engine": "native policy and native simulator; Python is recording-only",
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "native_extension": str(pathlib.Path(nightsim._engine.__file__).resolve()),
                    "native_extension_sha256": sha256(pathlib.Path(nightsim._engine.__file__).resolve())},
        "sources": {str(path): sha256(path) for path in sources},
    })

    started = time.perf_counter()
    frames = []
    history = []
    tick = 0
    while engine.agents() and engine.info()["time"] < args.seconds:
        stop_at = min(args.seconds, engine.info()["time"] + sim.dt)
        engine.run_policy(args.seconds, stop_at)
        info = engine.info()
        overlay = policy_overlay(engine)
        state = engine.state()
        near = current_near_bait(engine, overlay)
        measured = engine.evaluation()
        evaluation = {"agents": len(engine.agents()), "predators": len(engine.predators()),
                      "fruit": len(engine.fruits()), "score": info["score"],
                      "near_bait": near, "held30": measured["final_predators_near_bait_30s"],
                      "bait_present_estimated": overlay["bait"] is not None}
        events = [dict(kind=kind, time=when, agent=aid, age=age, energy=energy)
                  for kind, when, aid, age, energy in engine.pop_events()]
        frames.append(dict(tick=tick, time=info["time"], world=world(engine), policy=overlay,
                           input=state["observations"], actions=[], evaluation=evaluation,
                           native_events=events))
        if tick % 10 == 0:
            history.append(dict(time=info["time"], agents=evaluation["agents"],
                                predators=evaluation["predators"], score=info["score"]))
        if len(frames) == 100:
            with gzip.open(chunks / f"{tick // 100:05d}.json.gz", "wt", compresslevel=1) as stream:
                json.dump(frames, stream, separators=(",", ":"), allow_nan=False)
            frames.clear()
        tick += 1

    if frames:
        with gzip.open(chunks / f"{(tick - 1) // 100:05d}.json.gz", "wt", compresslevel=1) as stream:
            json.dump(frames, stream, separators=(",", ":"), allow_nan=False)
    info = engine.info()
    native_evaluation = engine.evaluation() if hasattr(engine, "evaluation") else None
    summary = dict(seed=args.seed, status="complete", frames=tick,
                   sim_time=info["time"], runtime_seconds=time.perf_counter() - started,
                   score=info["score"], final_agents=len(engine.agents()),
                   peak_agents=(native_evaluation or {}).get("peak_agents"),
                   predators=len(engine.predators()), history=history, events=[],
                   policy_metrics={"guide_assignments": (native_evaluation or {}).get("guide_attempts", 0),
                                   "delivery_arrivals": (native_evaluation or {}).get("guide_deliveries", 0),
                                   "overlapping_replacements": None},
                   estimated_bait_gap_seconds_after_first_arrival=(native_evaluation or {}).get("bait_gap_seconds_total", 0.0),
                   native_evaluation=native_evaluation,
                   metric_note="Truth is recorded for replay only. Held-30 proximity is not permanent capture.")
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"history", "events"}}, indent=2))


if __name__ == "__main__":
    main()
