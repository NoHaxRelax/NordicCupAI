"""Narrow predator-escape tests (nightsim hooks; tests only). One agent (typical late traits) + one fresh predator
facing it at distance D and bearing B (relative to the agent heading); run T seconds with the native policy.
usage: escape.py --configs CFG.json --seeds 1-16 --out rows.jsonl [--speeds 10 15 20] [--dists 40 80 150 250] [--bearings 0 90 180]
Row: label, seed, speed, D, B, killed (0/1), t_kill, energy_used, final_dist."""
import argparse, json, math, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds

def one(job):
    label, kw, seed, speed, D, B, T = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    k = eng.dbg_keep_agent(-1)
    if k is None: return None
    aid, x, y = k
    head = 0.3 * seed  # arbitrary heading
    e0 = 150.
    eng.dbg_set_agent(aid, x, y, head, e0, float(speed), float(min(40, 2 * speed)), 800., 100., 400., 1.57, 1000.)
    # predator at distance D, bearing B (deg) relative to the agent heading, facing the agent
    b = head + math.radians(B)
    px, py = x + D * math.cos(b), y + D * math.sin(b)
    if not eng.dbg_add_predator(px, py, b + math.pi, 200., False): return dict(label=label, seed=seed, speed=speed, D=D, B=B, skip=1)
    eng.policy_init(nightsim.seed_key(seed), kw)
    killed = 0; tk = None
    t0 = eng.info()['time']
    while eng.info()['time'] < t0 + T:
        eng.run_policy(t0 + T, eng.info()['time'] + 0.5)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'predator' and a == aid: killed = 1; tk = round(t - t0, 1)
        if killed or not eng.agents(): break
    ag = [a for a in eng.agents() if a[0] == aid]
    pr = eng.predators()
    fd = math.hypot(ag[0][1] - pr[0][0], ag[0][2] - pr[0][1]) if ag and pr else None
    used = (e0 - ag[0][5]) if ag else None
    return dict(label=label, seed=seed, speed=speed, D=D, B=B, killed=killed, t_kill=tk,
                used=used and round(used, 1), fdist=fd and round(fd), children=len(eng.agents()) - (1 if ag else 0))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--speeds', type=float, nargs='+', default=[10, 15, 20]); ap.add_argument('--dists', type=float, nargs='+', default=[40, 80, 150, 250])
    ap.add_argument('--bearings', type=float, nargs='+', default=[0, 90, 180]); ap.add_argument('--T', type=float, default=30.)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, sp, D, B, a.T) for l, kw in cfgs.items() for s in parse_seeds(a.seeds) for sp in a.speeds for D in a.dists for B in a.bearings]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=4):
            if r: f.write(json.dumps(r) + '\n'); n += 1
    print(f'done {n} scenarios in {time.time()-t0:.0f}s', flush=True)
