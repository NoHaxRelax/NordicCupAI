"""Map-structure check (nightsim hooks; engine truth only to grade): run a full game to --T, then compare the policy's
wall faces and crevice sites with the true map.
usage: mapcheck.py --configs CFG.json --seeds 1-32 --out rows.jsonl [--T 600] [--predators]
Row per game: label, seed, alive, n_faces, face_len, phantom_len (solid side actually free), miss_len (free side
actually blocked), by n_obs bucket, n_sites, sites_valid (grade: hold point free, predator cannot reach within 14.9,
lane free), top1_valid."""
import argparse, json, math, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
from nightsim.trapsite import grade

def one(job):
    label, kw, seed, T, preds = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=preds); eng = sim._engine
    sim.step([]); eng.pop_events()
    eng.policy_init(nightsim.seed_key(seed), dict(kw, trap_mode=max(1, int(kw.get('trap_mode', 0)))))
    while eng.info()['time'] < T and eng.agents():
        eng.run_policy(T, eng.info()['time'] + 5.)
        eng.pop_events()
    walls = eng.dbg_walls(); sites = eng.dbg_sites()
    def solid_free(x, y): return eng.dbg_free(x - 1, y - 1, 2)
    tot = ph = mi = 0.; buckets = {}
    for gid, horiz, c, lo, hi, solid, n, n_obs, conf, *_ in walls:
        if not conf or hi - lo < 4: continue
        L = hi - lo; k = max(2, int(L / 4)); bad_s = bad_f = 0
        for i in range(k + 1):
            a = lo + L * i / k
            xs, ys = (a, c + 3 * solid) if horiz else (c + 3 * solid, a)     # solid side: should be blocked
            xf, yf = (a, c - 3 * solid) if horiz else (c - 3 * solid, a)     # free side: should be free
            if solid_free(xs, ys): bad_s += 1
            if not solid_free(xf, yf): bad_f += 1
        tot += L; ph += L * bad_s / (k + 1); mi += L * bad_f / (k + 1)
        b = '1' if n_obs <= 1 else '2-3' if n_obs <= 3 else '4-9' if n_obs <= 9 else '10+'
        bb = buckets.setdefault(b, [0., 0.]); bb[0] += L; bb[1] += L * bad_s / (k + 1)
    valid = []
    for st in sites:
        gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = st
        f, b_, l = grade(eng, gx, gy, ox, oy)
        valid.append(int(f and b_ and l))
    return dict(label=label, seed=seed, alive=len(eng.agents()), t=round(eng.info()['time']), n_faces=len(walls), face_len=round(tot),
                phantom_len=round(ph), miss_len=round(mi), buckets={k: [round(v[0]), round(v[1])] for k, v in buckets.items()},
                n_sites=len(sites), sites_valid=sum(valid), top1_valid=valid[0] if valid else None)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--T', type=float, default=600.); ap.add_argument('--predators', action='store_true')
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, a.T, a.predators) for l, kw in cfgs.items() for s in parse_seeds(a.seeds)]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            if r: f.write(json.dumps(r) + '\n'); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
