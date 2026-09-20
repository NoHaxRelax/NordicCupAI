"""Where does the duplicate-action trick break? Measure each side's cost vs N.

Our side   : build N action dicts + json.dumps (this is what their timeout=10 read must cover).
Their side : json.loads + ActionRequest(**a) per action (simulation_server.py), then N agent_steps.
"""
import json, math, sys, time, resource

sys.path.insert(0, '/root/night/code/survival/vendor/survival-simulator')
PI = math.pi


def action(aid):
    return {"agent_id": aid, "move_distance": 0.0, "move_direction": 0.0, "turn_angle": PI, "spawn_agent": False}


def mb(s):
    return len(s) / 1e6


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


from src.utils.DTOs import ActionRequest

print(f"{'N':>9} {'build+dumps':>12} {'payload MB':>11} {'their loads':>12} {'their pydantic':>15} {'total our-side':>15} {'peak RSS MB':>12}")
for N in (20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000):
    t0 = time.perf_counter()
    acts = [action(7)] * N
    body = json.dumps({"actions": acts})
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    parsed = json.loads(body)["actions"]
    t_loads = time.perf_counter() - t0

    t0 = time.perf_counter()
    objs = [ActionRequest(**a) for a in parsed]
    t_pyd = time.perf_counter() - t0

    print(f"{N:>9} {t_build:>11.3f}s {mb(body):>10.2f} {t_loads:>11.3f}s {t_pyd:>14.3f}s {t_build:>14.3f}s {rss_mb():>11.0f}")
    del acts, body, parsed, objs
