"""Paired offline test of the harvest layer: same seeds, policy with and without it."""
import argparse, json, math, os, sys, time, pathlib
from multiprocessing import Pool
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
import nightsim
from harvest import Harvester


def one(job):
    label, cfg, seed, horizon, hv_kw = job
    t0 = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    state = sim.step([]); eng.policy_init(nightsim.seed_key(seed), dict(cfg))
    hv = Harvester(**hv_kw)
    eaten = fe = 0.; pen = 0.; pd = sd = 0; acts_max = 0; acts_tot = 0
    while state['observations'] and state['sim_time'] < horizon:
        base = [{"agent_id": a, "move_distance": d, "move_direction": di, "turn_angle": t, "spawn_agent": sp}
                for a, d, di, t, sp in eng.policy_act()]
        out = hv.apply(state['observations'], state['sim_time'], base)
        acts_max = max(acts_max, len(out)); acts_tot += len(out)
        state = sim.step([(a["agent_id"], a) for a in out])
        for kind, t, aid, age, energy in eng.pop_events():
            if kind == 'fruit': eaten += 1; fe += energy
            elif kind == 'predator': pd += 1; pen += energy / 100
            else: sd += 1
    info = eng.info()
    preds = eng.predators()
    return dict(label=label, seed=seed, surv=round(info['time'], 1), score=round(info['score'], 3),
                fruit=round(fe / 1000, 3), eaten=eaten, pdeaths=pd, sdeaths=sd, penalty=round(pen, 3),
                harvests=hv.harvests, hlog=hv.log, preds=len(preds),
                asleep=sum(1 for p in preds if p[4]), acts_max=acts_max, acts_mean=round(acts_tot / max(1, int(info['time'] * 10)), 1),
                wall=round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--seeds', required=True)
    ap.add_argument('--horizon', type=float, default=3000.); ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', required=True); ap.add_argument('--budget', type=int, default=20000)
    ap.add_argument('--max-harvests', type=int, default=8); ap.add_argument('--variants', default='base,harv')
    ap.add_argument('--selector', choices=('low_energy', 'old_chased', 'nearest_chased'), default='low_energy')
    ap.add_argument('--trigger', type=float, default=30.0); ap.add_argument('--closing', type=float, default=6.0)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    seeds = []
    for part in a.seeds.split(','):
        if '-' in part: x, y = part.split('-'); seeds += list(range(int(x), int(y) + 1))
        else: seeds.append(int(part))
    V = {'base': dict(enabled=False), 'harv': dict(enabled=True, budget=a.budget,
            max_harvests=a.max_harvests, sacrifice_mode=a.selector,
            trigger=a.trigger, closing=a.closing)}
    jobs = [(v, cfg, s, a.horizon, V[v]) for s in seeds for v in a.variants.split(',')]
    t0 = time.time()
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush()
    print(f'done {len(jobs)} runs in {time.time()-t0:.0f}s -> {a.out}', flush=True)
