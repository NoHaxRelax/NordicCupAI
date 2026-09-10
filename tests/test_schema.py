"""The reconstruction is only useful if its payload is shape-identical to theirs.

Every assertion here is against the captured response, so if the interface ever
drifts these fail loudly rather than silently training a policy against the wrong
observation format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survivalsim import ACTIONS, SurvivalSim, captured_spec, sample_spec  # noqa: E402

# Captured verbatim off their endpoint.
CAPTURED = json.loads("""
{
  "game_status": "running",
  "score": 123.4,
  "agent_status": [{
    "agent_id": 1,
    "observations": [
      {"type": "tree",     "distance": 12.5, "angle": 1.57},
      {"type": "predator", "distance": 30.0, "angle": -1.57, "rel_dir": -2.57},
      {"type": "edge",     "coords": [[50.0,50.0],[100.0,100.0]]}
    ],
    "energy": 85.0, "max_energy": 500.0,
    "biome": "forest", "age": 5.2,
    "speed": 12.5, "sprint_speed": 13.5,
    "hearing_radius": 10.0,
    "vision_angle": 1.57, "vision_range": 50.0
  }]
}
""")

ENTITY_FIELDS = {
    "tree": {"type", "distance", "angle"},
    "predator": {"type", "distance", "angle", "rel_dir"},
    "edge": {"type", "coords"},
}

FAILS: list[str] = []


def ck(name: str, cond: bool, extra: object = "") -> None:
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {extra}" if not cond and extra != "" else ""))
    if not cond:
        FAILS.append(name)


def collect(payload: dict) -> None:
    """Walk a payload far enough to have seen every entity type at least once."""
    ck("top-level keys match", set(payload) == set(CAPTURED), set(payload) ^ set(CAPTURED))
    ck("game_status is a known value", payload["game_status"] in ("running", "game_over"), payload["game_status"])
    ck("score is a float", isinstance(payload["score"], float), type(payload["score"]))
    ck("agent_status is a list", isinstance(payload["agent_status"], list))


def main() -> int:
    ref_agent = CAPTURED["agent_status"][0]

    # --- 1. the captured spec reproduces the captured shape
    sim = SurvivalSim(captured_spec(), seed=0)
    obs = sim.observe()
    collect(obs)

    a = obs["agent_status"][0]
    ck("agent keys match the capture", set(a) == set(ref_agent), set(a) ^ set(ref_agent))
    for k, v in ref_agent.items():
        if k == "observations":
            continue
        ck(f"agent field type: {k}", type(a[k]) is type(v), f"{type(a[k])} vs {type(v)}")
    ck("agent_id is 1-indexed", a["agent_id"] == 1, a["agent_id"])

    # --- 2. the captured constants survive the round trip
    for k in ("max_energy", "speed", "sprint_speed", "hearing_radius", "vision_angle", "vision_range", "biome"):
        ck(f"captured value preserved: {k}", a[k] == ref_agent[k], f"{a[k]} vs {ref_agent[k]}")
    ck("start energy matches capture", abs(a["energy"] - ref_agent["energy"]) < 1e-6, a["energy"])

    # --- 3. every entity type we emit is shaped like theirs
    seen: set[str] = set()
    rng = np.random.default_rng(7)
    for ep in range(60):
        sim = SurvivalSim(sample_spec(rng), seed=int(rng.integers(1 << 30)))
        for _ in range(120):
            payload = sim.observe()
            for ag in payload["agent_status"]:
                for e in ag["observations"]:
                    t = e["type"]
                    seen.add(t)
                    ck_once = set(e) == ENTITY_FIELDS[t]
                    if not ck_once:
                        ck(f"entity fields for {t}", False, set(e) ^ ENTITY_FIELDS[t])
                        return 1
            out = sim.step(rng.integers(0, len(ACTIONS), size=sim.spec.n_agents))
            if out[2]:
                break
    ck("all three entity types emitted", seen == set(ENTITY_FIELDS), seen)

    # --- 4. entity values are in the ranges the capture implies
    sim = SurvivalSim(captured_spec(), seed=3)
    bad_angle = bad_dist = 0
    for _ in range(400):
        for ag in sim.observe()["agent_status"]:
            for e in ag["observations"]:
                if e["type"] == "edge":
                    ck_c = (len(e["coords"]) == 2 and all(len(c) == 2 for c in e["coords"]))
                    if not ck_c:
                        ck("edge coords are two xy pairs", False, e["coords"])
                        return 1
                    continue
                if not (-np.pi <= e["angle"] <= np.pi):
                    bad_angle += 1
                if not (0.0 <= e["distance"] <= max(ag["vision_range"], ag["hearing_radius"]) + 1e-6):
                    bad_dist += 1
        if sim.step([1])[2]:
            sim.reset()
    ck("angles wrapped to (-pi, pi]", bad_angle == 0, bad_angle)
    ck("distances within vision or hearing", bad_dist == 0, bad_dist)

    # --- 5. the family is actually diverse, not a constant in disguise
    specs = [sample_spec(rng) for _ in range(400)]
    for f in ("drain_move", "tree_energy", "predator_aggro", "max_energy", "world_size"):
        vals = np.array([getattr(s, f) for s in specs])
        ck(f"randomised: {f}", vals.std() / (abs(vals.mean()) + 1e-9) > 0.15,
           f"cv={vals.std()/(abs(vals.mean())+1e-9):.3f}")
    n_ag = np.array([s.n_agents for s in specs])
    ck("multi-agent specs are sampled", (n_ag > 1).mean() > 0.5, f"{(n_ag>1).mean():.2f}")

    # --- 6. invariants under random play
    viol = []
    for ep in range(40):
        sim = SurvivalSim(sample_spec(rng), seed=int(rng.integers(1 << 30)))
        for _ in range(400):
            payload, rew, done, info = sim.step(rng.integers(0, len(ACTIONS), size=sim.spec.n_agents))
            if (sim.energy < -1e-9).any() or (sim.energy > sim.spec.max_energy + 1e-6).any():
                viol.append("energy out of bounds")
            if not np.isfinite(rew).all():
                viol.append("non-finite reward")
            if (sim.agent_xy < -1e-6).any() or (sim.agent_xy > sim.spec.world_size + 1e-6).any():
                viol.append("agent left the world")
            if done:
                break
    ck("no invariant violations under random play", not viol, set(viol))

    # --- 7. JSON round-trips (it has to survive going over the wire)
    ok = True
    try:
        json.loads(json.dumps(SurvivalSim(sample_spec(rng), seed=1).observe()))
    except (TypeError, ValueError) as exc:
        ok = False
        print("   ", exc)
    ck("payload is JSON-serialisable", ok)

    print()
    print(f"{len(FAILS)} FAILING: {FAILS}" if FAILS else "ALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
