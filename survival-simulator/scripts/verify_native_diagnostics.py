"""Development lockstep plus per-step native/Python accounting; no holdout or policy tuning."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', choices=('random', 'orchard'), default='random')
    args = parser.parse_args()
    if args.seed not in range(12) or not 1 <= args.steps <= 30000:
        parser.error('Use development seeds 0-11 and 1-30000 steps')
    # The verifier's counter-hash patch is scoped to this disposable process.
    from fastsim.verify import PySim, fastsim, RandomPolicy, _serial, first_diff, world_py, world_fast
    from scripts.research_diagnostics import Diagnostics
    _serial['c'] = itertools.count(1)
    py = PySim(seed=args.seed, starting_predators=2)
    fast = fastsim.SimulationCore(seed=args.seed, starting_predators=2)
    if args.policy == 'orchard':
        from fastsim.policy.orchard_ref import OrchardPolicy
        policy = OrchardPolicy(seed=args.seed)
    else:
        policy = RandomPolicy(args.seed)
    request = dict(case_id='native-diagnostic-check', seed=args.seed, policy_seed=0,
                   diagnostics=dict(screenshots=False))
    with tempfile.TemporaryDirectory() as folder:
        collectors = [Diagnostics(sim.env, Path(folder)/name, request, sim.dt)
                      for sim, name in ((py, 'python'), (fast, 'fastsim'))]
        actions, state, completed = [], None, 0
        try:
            for tick in range(args.steps):
                public = state['observations'] if state else []
                for d in collectors:
                    d.before(tick, public)
                    d.decision([(aid, act.model_dump()) for aid, act in actions], {})
                state = py.step(actions)
                other = fast.step(actions)
                for d in collectors:
                    d.after()
                diff = first_diff(state, other) or first_diff(world_py(py.env), world_fast(fast))
                if diff or py.env.rng.getstate() != fast._engine.rng_state():
                    raise AssertionError(diff or 'RNG divergence')
                for key in collectors[0].totals.keys() | collectors[1].totals.keys():
                    a, b = collectors[0].totals[key], collectors[1].totals[key]
                    if abs(a-b) > 1e-8*max(1, abs(a), abs(b)):
                        raise AssertionError(f'{key}: Python {a} != native {b}')
                if collectors[0].death_causes != collectors[1].death_causes:
                    raise AssertionError('Death cause mismatch')
                if collectors[0].agents.keys() != collectors[1].agents.keys():
                    raise AssertionError('Agent lifetime IDs differ')
                for aid, left in collectors[0].agents.items():
                    right = collectors[1].agents[aid]
                    for key in ('parent', 'born', 'last_meal', 'death'):
                        if left[key] != right[key]:
                            raise AssertionError(f'Agent {aid} {key} mismatch')
                    for key in left['ledger'].keys() | right['ledger'].keys():
                        a, b = left['ledger'][key], right['ledger'][key]
                        if abs(a-b) > 1e-8*max(1, abs(a), abs(b)):
                            raise AssertionError(f'Agent {aid} {key}: {a} != {b}')
                completed = tick+1
                if not state['num_agents']:
                    break
                actions = policy(state['observations'], state['sim_time'])
        finally:
            summaries = [d.finish('extinct' if not sim.env.agents else 'interrupted')
                         for d, sim in zip(collectors, (py, fast))]
    print(json.dumps(dict(ok=True, steps=completed, seed=args.seed, policy=args.policy, totals=summaries[0]['totals'],
                          deaths=summaries[0]['deaths'], ordering='creation-counter')))


if __name__ == '__main__':
    main()
