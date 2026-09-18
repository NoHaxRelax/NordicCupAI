"""No-predator variant of the society harness.

Assumes predator control is solved elsewhere: `Environment.spawn_predator` is
replaced by a no-op before the environment is built, so no predator ever
exists. Everything else (engine, diagnostics, output format) is the society
harness unchanged. Local research only; no online submissions.
"""
import sys, pathlib, json, argparse
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'society'))
sys.path.insert(0, str(ROOT / 'research' / 'orchard'))
sys.path.insert(0, str(ROOT / 'debugger'))   # ReplayRecorder for --record
import harness  # noqa: E402  (sets up vendor path)
from src.elements.environment import Environment  # noqa: E402

Environment.spawn_predator = lambda self, *a, **k: None  # no predators, ever

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='orchard:OrchardPolicy')
    p.add_argument('--seeds', type=int, nargs='+', default=[1])
    p.add_argument('--horizon', type=float, default=3000)
    p.add_argument('--out', default=str(ROOT / 'results' / 'orchard'))
    p.add_argument('--label', default='')
    p.add_argument('--record', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--kw', default='{}')
    p.add_argument('--world-log', default=None)
    p.add_argument('--world-every', type=float, default=5.)
    a = p.parse_args()
    for seed in a.seeds:
        wl = None if a.world_log is None else (a.world_log if len(a.seeds) == 1 else f'{a.world_log}.seed{seed}')
        harness.run(a.policy, seed, a.horizon, a.out, a.label, a.record, a.verbose,
                    world_log=wl, world_every=a.world_every, **json.loads(a.kw))
