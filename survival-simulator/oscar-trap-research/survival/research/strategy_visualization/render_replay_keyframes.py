"""Render saved replay states with the upstream simulator renderer.

These are offline reconstructions, not footage captured during the original run.
No simulation steps or policy decisions are executed.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "research"), str(ROOT / "debugger"), str(ROOT / "vendor" / "survival-simulator")]

from predator_control import controlled_env
from recorder import NativeRenderer
from src.elements.agent import Agent
from src.elements.obstacle import Obstacle
from src.elements.predator import Predator


def nearest_frame(frames, wanted):
    return min(frames, key=lambda frame: abs(frame["t"] - wanted))


def apply_common(entity, row):
    for key in ("x", "y", "direction", "size", "energy", "max_energy", "speed", "sprint_speed", "hearing_radius"):
        if key in row:
            setattr(entity, key, row[key])
    if "vision_range" in row:
        entity.vision_radius = row["vision_range"]
    if "vision_angle" in row:
        entity.cone_angle = row["vision_angle"]


def install_frame(env, frame):
    env.time = frame["t"]
    env.score = frame.get("score", 0)
    agents = []
    for row in frame["agents"]:
        agent = Agent(row["x"], row["y"], rng=random.Random(int(row["id"]) + 1000))
        agent.agent_id = row["id"]
        apply_common(agent, row)
        agent.age = row.get("age", 0)
        agent.max_age = row.get("max_age", 120)
        agents.append(agent)
    predators = []
    for row in frame["predators"]:
        predator = Predator(row["x"], row["y"], rng=random.Random(int(row["id"]) + 2000))
        apply_common(predator, row)
        predator.resting = bool(row.get("resting", False))
        predators.append(predator)
    env.agents = agents
    env.agents_dict = {agent.agent_id: agent for agent in agents}
    env.predators = predators
    env.fruits = []
    env.trees = []
    env._update_spatial_grid()


def render_job(replay, stem, times):
    with gzip.open(replay, "rt", encoding="utf8") as source:
        data = json.load(source)
    env = controlled_env()
    env.obstacles = [Obstacle(item["x"], item["y"], item["width"], item["height"]) for item in data["world"]["obstacles"]]
    env.edges = {tuple(sorted(edge)) for obstacle in env.obstacles for edge in obstacle.edges}
    env._update_spatial_grid()
    renderer = NativeRenderer(env, width=800)
    written = []
    for index, wanted in enumerate(times):
        frame = nearest_frame(data["frames"], wanted)
        install_frame(env, frame)
        encoded = renderer.capture().split(",", 1)[1]
        target = HERE / "keyframes" / f"{stem}-{index:02d}-{frame['t']:.1f}.png"
        import base64
        target.write_bytes(base64.b64decode(encoded))
        written.append({"time": frame["t"], "file": target.name})
    return written


def extract_native_job(replay, stem, times):
    import base64
    with gzip.open(replay, "rt", encoding="utf8") as source:
        data = json.load(source)
    written = []
    for index, wanted in enumerate(times):
        frame = nearest_frame(data["frames"], wanted)
        encoded = frame["native_image"].split(",", 1)[1]
        target = HERE / "keyframes" / f"{stem}-{index:02d}-{frame['t']:.1f}.png"
        target.write_bytes(base64.b64decode(encoded))
        written.append({"time": frame["t"], "file": target.name})
    return written


JOBS = [
    (
        ROOT / "results/intake_guides_sol/replays/observed-gap-guides-v4-aligned-hold-g15-l90x70-o-5-h1-n33-i90-lat60-s4331-e830bf1d.json.gz",
        "guided-v4-success-4331",
        [0, 6.5, 996.5, 1986.5, 2886.5, 3000],
    ),
    (
        ROOT / "results/intake_validation/guided_gap_v1/replays/observed-gap-guides-g15-n33-i90-lat30-s442-f0219a47.json.gz",
        "guided-v1-success-442",
        [0, 6.5, 996.5, 1986.5, 2886.5, 3000],
    ),
    (
        ROOT / "results/intake_validation/guided_gap_v1/replays/observed-gap-guides-g15-n33-i90-lat90-s441-21934612.json.gz",
        "guided-v1-fail-441",
        [0, 6.3, 2700, 2910, 2939.8, 3000],
    ),
    (
        ROOT / "results/intake_validation/guided_gap_v2/replays/observed-gap-guides-v2-g15.3263-l86.9616x79.0562-o8.46975-h1-n33-i90-lat60-s462-3021f604.json.gz",
        "guided-v2-fail-462",
        [0, 6.5, 900, 1800, 2700, 3000],
    ),
    (
        ROOT / "results/intake_gate_sol/replays/intake-gate-sol-v3-n3-i90-s452-a1-05dce82d.json.gz",
        "single-wall-fallback-452",
        [0, 1.3, 91.3, 181.3, 900, 3000],
    ),
]

NATIVE_JOBS = [
    (
        ROOT / "results/wall_funneling/replays/observed-w30-l100-n33-sites1-s62-spread15.0-waves1-gate1-h0-a1-2f24d447.json.gz",
        "single-wall-group-native-62",
        [0, 12, 30, 60, 120, 180],
    ),
    (
        ROOT / "results/intake_geometry_sol/replays/observed-gap-g15-l90x70-o-5-h1-n33-i3-s306-88037ef6.json.gz",
        "direct-unequal-gap-native-306",
        [0, 12, 36, 72, 120, 180],
    ),
    (
        ROOT / "results/intake_validation/gap_v1/replays/observed-gap-g15-l90-n33-i90-s413-747ab40d.json.gz",
        "direct-gap-full-native-413",
        [0, 906, 1806, 2706, 2910, 3000],
    ),
    (
        ROOT / "results/intake_validation/guided_gap_v2/replays/observed-gap-guides-v2-g14.9669-l81.7861x95.5125-o-14.7551-h0-n33-i90-lat90-s461-ad144823.json.gz",
        "guided-v2-native-fail-461",
        [0, 906, 1806, 2706, 2910, 3000],
    ),
]


if __name__ == "__main__":
    if Path("/tmp/predator-intake-stop").exists():
        raise SystemExit("stop sentinel present")
    manifest = {}
    for replay, stem, times in JOBS:
        manifest[stem] = render_job(replay, stem, times)
    for replay, stem, times in NATIVE_JOBS:
        manifest[stem] = extract_native_job(replay, stem, times)
    (HERE / "keyframes" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
