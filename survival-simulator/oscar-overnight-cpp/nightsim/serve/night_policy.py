"""API adapter for the nightsim C++ policy (incl. the st_* NaN-statue layer): agent_status dicts -> actions."""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))   # parent of nightsim/
import nightsim
from nightsim import _nengine

TYPES = {"fruit": 0, "agent": 1, "predator": 2, "tree": 3, "edge": 4}
BIOMES = {"forest": 0, "swamp": 1, "desert": 2, "grassland": 3, "river": 4}


def convert(states):
    out = []
    for s in states:
        obs = []
        for o in s.get("observations", []):
            ty = TYPES[str(o["type"]).lower()]
            if ty == 4:
                (x1, y1), (x2, y2) = o["coords"]
                obs.append((4, 0., 0., None, None, float(x1), float(y1), float(x2), float(y2)))
            else:
                rd = o.get("rel_dir"); oid = o.get("id")
                obs.append((ty, float(o["distance"]), float(o["angle"]), None if rd is None else float(rd),
                            None if oid is None else int(oid), 0., 0., 0., 0.))
        out.append((int(s["agent_id"]), float(s["energy"]), BIOMES[str(s["biome"]).lower()], float(s["age"]),
                    float(s["speed"]), float(s["sprint_speed"]), float(s["hearing_radius"]), float(s["vision_angle"]),
                    float(s["vision_range"]), float(s["max_energy"]), obs))
    return out


class NightPolicy:
    def __init__(self, config, seed=0):
        self.eng = _nengine.Engine(nightsim.seed_key(seed), predators=False)
        self.eng.policy_init(nightsim.seed_key(seed), dict(config))

    def raw(self, states, sim_time):
        return self.eng.policy_act_ext(convert(states), float(sim_time))

    def __call__(self, states, sim_time):
        return [{"agent_id": a, "move_distance": d, "move_direction": di, "turn_angle": t, "spawn_agent": bool(sp)}
                for a, d, di, t, sp in self.raw(states, sim_time)]
