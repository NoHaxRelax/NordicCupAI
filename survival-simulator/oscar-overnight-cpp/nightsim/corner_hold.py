"""Why do predators leave the teleport corner? (tests only; engine truth)
Runs ins_or10-style games (prisoner insurance + every predator moved to the pin zone 10 s after spawn), samples every
tick, and for each predator records pin episodes (within 70 of (1595,1195)) and the state at each exit:
prisoners alive (agents at the corner), their energies, nearest free agent distance to the predator, predator resting,
predator energy, predator position just before/after. Row per exit."""
import argparse, json, math, sys, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
CX, CY = 1595., 1195.

def one(job):
    seed, kw, T = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([])
    kw = dict(kw); eng.dbg_corner_oracle(float(kw.pop('test_corner_oracle', 0.))); eng.dbg_pred_life(0.)
    eng.policy_init(nightsim.seed_key(seed), kw)
    pinned = {}; exits = []; entries = 0
    t = 0.
    while eng.agents() and t < T:
        eng.run_policy(T, t + 0.1); t = eng.info()['time']
        ags = eng.agents(); prs = eng.predators()
        pris = [a for a in ags if abs(a[1] - CX) < 1e-6 and abs(a[2] - CY) < 1e-6]
        free = [a for a in ags if not (abs(a[1] - CX) < 1e-6 and abs(a[2] - CY) < 1e-6)]
        for i, p in enumerate(prs):
            d = math.hypot(p[0] - CX, p[1] - CY)
            if d < 70.:
                if i not in pinned: pinned[i] = t; entries += 1
            elif i in pinned:
                t0 = pinned.pop(i)
                nf = min((math.hypot(a[1] - p[0], a[2] - p[1]) for a in free), default=1e9)
                exits.append(dict(t=round(t, 1), held=round(t - t0, 1), n_pris=len(pris), pris_e=sorted(round(a[5]) for a in pris)[-3:],
                                  near_free=round(nf), rest=bool(p[4]), pe=round(p[3]), pos=(round(p[0]), round(p[1])), d=round(d)))
    return dict(seed=seed, surv=round(t, 1), entries=entries, still=len(pinned), exits=exits)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--T', type=float, default=3000.); ap.add_argument('--workers', type=int, default=24); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    kw = json.load(open(a.config))
    with Pool(a.workers) as pool, open(a.out, 'w') as f:
        for r in pool.imap_unordered(one, [(s, kw, a.T) for s in parse_seeds(a.seeds)]):
            f.write(json.dumps(r) + '\n'); f.flush()
    print('done', flush=True)
