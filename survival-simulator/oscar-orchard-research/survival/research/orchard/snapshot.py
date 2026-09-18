"""Late-game checkpoints: run a seed to `t_snap` under a base configuration, pickle engine + controller,
then resume many late-game experiments from that state. Same engine, same rules; only the expensive
early game is shared. pygame surfaces are dropped before pickling and rebuilt as blanks on load
(they are rendering-only: no engine rule reads them).

  snapshot.py make --seeds 1 2 3 --t 1500 --out DIR [--kw JSON]
  snapshot.py run  --snap DIR/seed1.pkl.gz --kw JSON --label L --out RESULTS_DIR
"""
import os, sys, pathlib, json, argparse, gzip, pickle, time, copy
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
sys.path.insert(0, str(ROOT/'research'/'society'))
import pygame
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy

SURFACES = ('biome_surface', 'static_surface', 'shadow_surface', 'obstacle_surface', 'world_surface', 'vision_screen', 'leaf_screen')


def strip(env):
    for k in SURFACES: setattr(env, k, None)
    for tree in env.trees: pass
    return env


def restore(env):
    for k in SURFACES: setattr(env, k, pygame.Surface((env.width, env.height), pygame.SRCALPHA))
    return env


def make(seed, t_snap, kw, out):
    pygame.init()
    sim = SimulationCore(seed=seed); pol = OrchardPolicy(seed=seed, **kw)
    state = sim.step([]); t0 = time.perf_counter()
    while state['num_agents'] and state['sim_time'] < t_snap:
        state = sim.step(pol(state['observations'], state['sim_time']))
    if not state['num_agents']: print(f"seed {seed}: colony died at {state['sim_time']:.0f} before the snapshot"); return
    env = sim.env; strip(env)
    blob = dict(seed=seed, t=state['sim_time'], base_kw=kw, env=env, rng=sim.rng, policy=pol, state=state,
                dt=sim.dt, alive=state['num_agents'], trees=len(env.trees), score=state['score'])
    path = pathlib.Path(out)/f'seed{seed}.pkl.gz'; path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as f: pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    restore(env)
    print(f"seed {seed}: snapshot at {state['sim_time']:.0f} s, {state['num_agents']} agents, {len(env.trees)} trees, score {state['score']:.0f}, {time.perf_counter()-t0:.0f}s wall, {path.stat().st_size//1024} KB")


def run(snap, kw, label, out, horizon=3000., trace=0.):
    pygame.init()
    with gzip.open(snap, 'rb') as f: blob = pickle.load(f)
    env = restore(blob['env']); pol = blob['policy']; state = blob['state']
    # parameters added after the checkpoint was taken get their defaults; then apply the late-game overrides
    for k, v in OrchardPolicy(seed=0).P.items(): pol.P.setdefault(k, v)
    pol.P.update(kw)
    sim = SimulationCore.__new__(SimulationCore); sim.env = env; sim.rng = blob['rng']; sim.dt = blob['dt']; sim.seed = blob['seed']
    t0 = time.perf_counter(); samples = []; nxt = (int(state['sim_time']//100)+1)*100; tnext = 2000.
    import math
    while state['num_agents'] and state['sim_time'] < horizon:
        if trace and state['sim_time'] >= tnext-1e-6:
            tnext += trace; t = state['sim_time']
            print(f"== t={t:.0f} agents={len(env.agents)} trees={len(env.trees)} fruits={len(env.fruits)}")
            for ag in sorted(env.agents, key=lambda x: x.agent_id):
                m = pol.minds.get(ag.agent_id); g = pol.groups[m.group] if m else None
                dtree = min((math.dist((ag.x, ag.y), (tr.x, tr.y)) for tr in env.trees), default=None)
                dfruit = min((math.dist((ag.x, ag.y), (f.x, f.y)) for f in env.fruits), default=None)
                known = sum(1 for tr in g.trees.values() if not tr.dead) if g else '-'
                print(f"  a{ag.agent_id} {pol.decisions.get(ag.agent_id, ('?',))[0]:<8} e={ag.energy:5.0f} age={ag.age:4.0f} maxAge={ag.max_age:.0f} tree={dtree and round(dtree)} fruit={dfruit and round(dfruit)} knownLive={known} post={m.post if m else '-'} heir={m.heir_done if m else '-'}")
        state = sim.step(pol(state['observations'], state['sim_time']))
        if state['sim_time'] >= nxt-1e-6:
            nxt += 100; samples.append(dict(time=round(state['sim_time'], 1), alive=state['num_agents'], trees=len(env.trees), fruits=len(env.fruits), score=round(state['score'], 2)))
    T = state['sim_time']; last = samples[-1] if samples else dict(trees=len(env.trees), fruits=len(env.fruits))
    wall = T < horizon-1e-6 and last['trees'] <= 2 and last['fruits'] <= 5
    res = dict(label=label, seed=blob['seed'], from_t=blob['t'], survival=round(T, 1), score=round(state['score'], 3), kw=kw, base_kw=blob['base_kw'],
               trees_death=last['trees'], fruits_death=last['fruits'], wall_death=wall, eff_survival=horizon if wall else T,
               samples=samples, wall_seconds=round(time.perf_counter()-t0, 1))
    outp = pathlib.Path(out); outp.mkdir(parents=True, exist_ok=True)
    (outp/f'{label}-seed{blob["seed"]}.json').write_text(json.dumps(res)+'\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('samples', 'kw', 'base_kw')}), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd')
    m = sub.add_parser('make'); m.add_argument('--seeds', type=int, nargs='+', required=True); m.add_argument('--t', type=float, default=1500.)
    m.add_argument('--out', required=True); m.add_argument('--kw', default='{}'); m.add_argument('--workers', type=int, default=1)
    r = sub.add_parser('run'); r.add_argument('--snap', required=True); r.add_argument('--kw', default='{}'); r.add_argument('--label', default='late')
    r.add_argument('--out', required=True); r.add_argument('--trace', type=float, default=0.)
    a = ap.parse_args()
    if a.cmd == 'make':
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers) as pool: pool.starmap(make, [(s, a.t, json.loads(a.kw), a.out) for s in a.seeds])
        else:
            for s in a.seeds: make(s, a.t, json.loads(a.kw), a.out)
    else:
        run(a.snap, json.loads(a.kw), a.label, a.out, trace=a.trace)
