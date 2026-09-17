"""Full generated-map games: society alone vs society + trapper (oracle world).

    ../.venv/bin/python scripts/trapper/run_game.py --seeds 1 2 3 --seconds 900 --mode both --record

Writes one JSON per run under results/trapper/ (metrics, events, samples) and,
with --record, a replay into Oscar's Survival Lab results folder. Read-only
diagnostics wrap kill_agent to label deaths; the engine is unchanged.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import SimulationCore                                   # noqa: E402
from models.trapper.policy import TrapperPolicy                        # noqa: E402
from models.trapper.recording import make_recorder, save_recorder      # noqa: E402

OUT = ROOT / 'results' / 'trapper'


def run(seed, seconds, trap, record=False, native=False, label='', verbose=False, params=None, society_kwargs=None):
    start = time.perf_counter()
    sim = SimulationCore(seed=seed)
    env = sim.env
    policy = TrapperPolicy(seed=seed, env=env, trap=trap, society_kwargs=society_kwargs, **(params or {}))
    deaths = []
    phase = {'pred': False}
    orig_kill = env.kill_agent
    orig_step = env.non_agent_step

    def kill_agent(agent):
        deaths.append(dict(t=round(env.time, 1), id=agent.agent_id, age=round(agent.age, 1), energy=round(agent.energy, 1),
                           cause='predator' if phase['pred'] else 'starvation',
                           role=policy.manager.role_of(agent.agent_id) if trap else None))
        return orig_kill(agent)

    def non_agent_step(dt):
        phase['pred'] = False
        preds = env.predators

        class Watch(list):
            def __iter__(inner):
                phase['pred'] = True
                return list.__iter__(inner)
        env.predators = Watch(preds)
        try:
            return orig_step(dt)
        finally:
            new = list(env.predators)
            env.predators = preds
            preds[:] = new
            phase['pred'] = False
    env.kill_agent, env.non_agent_step = kill_agent, non_agent_step

    mode = 'trapper' if trap else 'society'
    rec = make_recorder(env, title=f'{mode} seed {seed} {label}'.strip(), policy=f'trapper policy ({mode}, oracle world)', seed=seed,
                        scenario='generated', notes='society baseline (Oscar v11+) with trap manager overrides; oracle world state',
                        native=native, every=1 if native else 5) if record else None
    state = sim.step([])
    samples = []
    next_sample = 30.0
    peak = state['num_agents']
    held_hist = []
    while state['num_agents'] and state['sim_time'] < seconds:
        obs = state['observations']
        t0 = state['sim_time']
        actions = policy(obs, t0)
        state = sim.step(actions)
        if rec:
            rec.capture(actions, policy.last_decisions, obs, t0)
        peak = max(peak, state['num_agents'])
        if trap:
            held = sum(len(st.held) for st in policy.manager.stations.values())
            held_hist.append((len(env.predators), held))
        if state['sim_time'] >= next_sample - 1e-6:
            next_sample += 30.0
            row = dict(t=round(state['sim_time'], 1), score=round(state['score'], 2), alive=state['num_agents'],
                       predators=len(env.predators), trees=len(env.trees), fruits=len(env.fruits))
            if trap:
                row['held'] = sum(len(st.held) for st in policy.manager.stations.values())
                row['roles'] = {r: sum(1 for v in policy.manager.roles.values() if v[0] == r) for r in ('bait', 'guide', 'successor', 'guard')}
                row['deliveries'] = len(policy.manager.deliveries)
            samples.append(row)
            if verbose:
                print(json.dumps(row), flush=True)
    pred_deaths = [d for d in deaths if d['cause'] == 'predator']
    result = dict(mode=mode, seed=seed, horizon=seconds, label=label, survival=round(state['sim_time'], 1), score=round(state['score'], 3),
                  alive=state['num_agents'], peak=peak, predators=len(env.predators),
                  predator_deaths=len(pred_deaths), starvation_deaths=len(deaths) - len(pred_deaths),
                  predator_penalty=round(sum(d['energy'] for d in pred_deaths) / 100, 2),
                  role_deaths={r: sum(1 for d in pred_deaths if d['role'] == r) for r in ('bait', 'guide', 'successor', 'guard', None)} if trap else None,
                  held_fraction=round(sum(h for _, h in held_hist) / max(1, sum(n for n, _ in held_hist)), 3) if trap else None,
                  metrics=policy.metrics if trap else policy.society.metrics,
                  events=policy.manager.events if trap else [], deaths=deaths, samples=samples,
                  wall_seconds=round(time.perf_counter() - start, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    name = f'{mode}-seed{seed}{("-" + label) if label else ""}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}'
    (OUT / f'{name}.json').write_text(json.dumps(result, indent=1))
    if rec:
        result['replay'] = str(save_recorder(rec, f'game-{mode}-seed{seed}{("-" + label) if label else ""}', reason='horizon' if state['num_agents'] else 'extinct'))
    brief = {k: result[k] for k in ('mode', 'seed', 'survival', 'score', 'alive', 'peak', 'predators', 'predator_deaths', 'starvation_deaths', 'predator_penalty', 'held_fraction', 'wall_seconds')}
    if trap:
        brief['trap'] = {k: v for k, v in policy.manager.metrics.items()}
        brief['role_deaths'] = result['role_deaths']
    print(json.dumps(brief), flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1])
    ap.add_argument('--seconds', type=float, default=900)
    ap.add_argument('--mode', choices=['society', 'trapper', 'both'], default='both')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--label', default='')
    ap.add_argument('--params', default='{}', help='JSON overrides for the trap manager')
    a = ap.parse_args()
    params = json.loads(a.params)
    rows = []
    for seed in a.seeds:
        if a.mode in ('society', 'both'):
            rows.append(run(seed, a.seconds, False, a.record, a.native, a.label, a.verbose))
        if a.mode in ('trapper', 'both'):
            rows.append(run(seed, a.seconds, True, a.record, a.native, a.label, a.verbose, params))
    if a.mode == 'both':
        soc = [r for r in rows if r['mode'] == 'society']
        tr = [r for r in rows if r['mode'] == 'trapper']
        print(f"society mean score {sum(r['score'] for r in soc)/len(soc):.1f}, survival {sum(r['survival'] for r in soc)/len(soc):.1f}")
        print(f"trapper mean score {sum(r['score'] for r in tr)/len(tr):.1f}, survival {sum(r['survival'] for r in tr)/len(tr):.1f}")
