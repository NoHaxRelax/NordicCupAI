"""Full games with stuck-spot guiding (sg_*): the policy gets the probe's sinks for its seed (engine-truth ceiling), predators on.
usage: sggame.py --configs CFG --sinks probe.jsonl --seeds 1-192 --out rows.jsonl [--minn 3] [--T 3000]
Row: label, seed, surv, score, pdeaths, sg counters [episodes, reached, died_at_sink, died_en_route, lost, escaped, timeout, waited],
traj every 250 s: [t, alive, predators, trapped300 (stationary within 30 for >= 300 s), trapped120, at_sink (within 30 of a sink vertex)],
trapped_end (stationary >= 300 s at the end), trapped_end120."""
import argparse, json, math, os, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
from nightsim.stuckspot import Tracker

SINKS = {}
SPECS = {}
MINROB = float(os.environ.get('SG_MINROB', '0')); ORACLE = int(os.environ.get('SG_ORACLE', '0')); ORACLE_T = float(os.environ.get('SG_ORACLE_T', '900')); ORACLE_MINN = int(os.environ.get('SG_ORACLE_MINN', '3')); ORACLE_NS = int(os.environ.get('SG_ORACLE_NS', '3')); MINKILL = float(os.environ.get('SG_MINKILL', '0.75'))


def sink_tuple(sp_, minkill=0.75, k=5):
    """policy sink tuple from a spec row [si, vx, vy, n, ok, cands]: (vx, vy, [Gx, Gy, th, Lx, Ly] * <= k), cands with a
    kill-test score >= minkill (the best one always), one per heading"""
    out = [sp_[1], sp_[2]]; seen = set()
    for i, c in enumerate(sp_[5]):
        if i > 0 and (len(c) < 7 or c[6] < minkill): continue
        if c[2] in seen: continue
        seen.add(c[2]); out += [c[0], c[1], c[2], c[4], c[5]]
        if len(seen) >= k: break
    return tuple(out)


def one(job):
    label, kw, seed, T, minn = job
    import nightsim
    t0w = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([]); eng.pop_events()
    if SPECS:
        sp_ = [x for x in SPECS.get(seed, []) if x[5] and x[5][0][3] >= MINROB]
        sinks = [sink_tuple(x, MINKILL) for x in sp_ if len(x[5][0]) < 7 or x[5][0][6] >= MINKILL]
    else:
        sinks = [(s[0], s[1]) for s in SINKS.get(seed, []) if s[2] >= minn]
    eng.policy_init(nightsim.seed_key(seed), kw)
    eng.dbg_sg_sinks(sinks)
    if kw.get('sg_clear', 0) > 0: eng.dbg_sg_rects()
    tr = Tracker(); pd = 0; traj = []; nxt = 250.; moved = {}
    osinks = [x for x in SINKS.get(seed, []) if len(x) > 4 and x[2] >= ORACLE_MINN][:ORACLE_NS]
    while True:
        info = eng.info()
        if not eng.agents() or info['time'] >= T: break
        eng.run_policy(T, info['time'] + 1.0)
        for kind, t, aid, age, en in eng.pop_events():
            if kind == 'predator': pd += 1
        tn = eng.info()['time']; P = eng.predators()
        if ORACLE > 0 and tn <= ORACLE_T and osinks:   # oracle delivery: the first ORACLE predators go straight into a cycle state
            for i in range(min(len(P), ORACLE)):
                if i in moved: continue
                sk = osinks[len(moved) % len(osinks)]
                if eng.dbg_move_predator(i, sk[0], sk[1], sk[4], 150.): moved[i] = tn
            P = eng.predators()
        tr.update(tn, P, eng.dbg_pred_info(), [])
        if tn >= nxt - 1e-6:
            tp = [tn - s['t0'] for s in tr.st.values()]
            at = sum(1 for p in P if any(math.hypot(p[0] - s[0], p[1] - s[1]) < 30 for s in sinks))
            ags = eng.agents()
            sd = [round(min((math.hypot(a[1] - s[0], a[2] - s[1]) for a in ags), default=9999)) for s in sinks]
            traj.append([int(nxt), len(ags), len(P), sum(1 for x in tp if x >= 300), sum(1 for x in tp if x >= 120), at, sd])
            nxt += 250.
    t = eng.info()['time']
    tp = [t - s['t0'] for s in tr.st.values()]
    cnt, per = eng.dbg_sg_sinks()
    return dict(label=label, seed=seed, surv=round(t, 1), score=round(eng.info()['score'], 3), pdeaths=pd, sg=cnt, nsinks=len(sinks),
                sent=[x[0] for x in per[:-1]], diag=per[-1], preds=len(eng.predators()), trapped_end=sum(1 for x in tp if x >= 300),
                trapped_end120=sum(1 for x in tp if x >= 120), moved=len(moved), traj=traj, wall=round(time.perf_counter() - t0w, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--sinks', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--minn', type=int, default=3); ap.add_argument('--T', type=float, default=3000.)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    for l in open(a.sinks):
        r = json.loads(l)
        if r.get('mode') == 'spec': SPECS[r['seed']] = r['specs']
        else: SINKS[r['seed']] = r['sinks']
    jobs = [(l, kw, s, a.T, a.minn) for s in parse_seeds(a.seeds) for l, kw in cfgs.items()]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
