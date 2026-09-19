"""Stuck-spot guiding scenario (sg_mode): one guide leads one predator into a sink; engine truth only to set up and grade.
usage: sgscen.py --configs CFG --sinks probe.jsonl --seeds 1-64 --out rows.jsonl [--dg 120 250] [--dp 90 140] [--bear 0 90 180]
       [--speeds 10 14] [--minn 3] [--top 3] [--T 120] [--hold 300]
Per (seed, sink among the top --top probe sinks with >= --minn captures, dg, dp, bear, speed): fresh map without predators,
keep one agent (the guide; NIGHT_NBY extra bystanders 40-160 from it), true poses, the policy gets ALL probe sinks with
>= --minn captures (dbg_sg_sinks). Guide dg from the sink (free spot, straight line not required), predator dp from the guide
at bearing `bear` (0 = behind the guide, i.e. away from the sink; 180 = between guide and sink), awake, facing the guide.
Run the policy up to T s (while agents live), then --hold s more (engine only once no agent is left). Row: delivered
(predator within R of the sink for the whole final --hold s), pred_end_d (final predator-sink distance), guide fate
(died / alive), t_guide_died, guide distance to the sink at death, sg counters."""
import argparse, json, math, os, sys, time, pathlib, random
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds

R = float(os.environ.get('STUCK_R', '30'))
NBY = int(os.environ.get('NIGHT_NBY', '0'))
GE = float(os.environ.get('NIGHT_GE', '300'))
TRACE = int(os.environ.get('NIGHT_TRACE', '0'))
SINKS = {}
SPECS = {}
LINE = int(os.environ.get('NIGHT_LINE', '1'))
MULTI = int(os.environ.get('SG_MULTI', '1')); MINKILL = float(os.environ.get('SG_MINKILL', '0.75'))


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
    label, kw, seed, si, dg, dp, bear, sp, T, HOLD, minn = job
    import nightsim
    if SPECS:
        sp_ = SPECS[seed][si]; sx, sy = sp_[1], sp_[2]; G = sp_[5][0]
        pol_sink = sink_tuple(sp_, MINKILL) if MULTI else (sx, sy, G[0], G[1], G[2], G[4], G[5]); ax, ay = G[4], G[5]
        base = dict(label=label, seed=seed, sink=sp_[0], spec=G, spec_ok=sp_[4], spec_n=sp_[3], dg=dg, dp=dp, bear=bear, speed=sp)
    else:
        sinks = [s for s in SINKS[seed] if s[2] >= minn]
        sx, sy = sinks[si][0], sinks[si][1]; pol_sink = (sx, sy); ax, ay = sx, sy
        base = dict(label=label, seed=seed, sink=si, sink_n=sinks[si][2], sink_info=sinks[si][3:], dg=dg, dp=dp, bear=bear, speed=sp)
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    ags = eng.dbg_keep_agents(1 + NBY)
    if len(ags) < 1 + NBY: return dict(base, skip='agents')
    guide = ags[0][0]; bys = [a[0] for a in ags[1:]]
    rng = random.Random(seed * 1000 + si * 97 + int(dg) + int(dp) * 7 + int(bear) * 13 + int(sp))
    found = None
    def line_ok(x0, y0, x1, y1):
        n_ = max(2, int(math.hypot(x1 - x0, y1 - y0) / 6))
        return all(not eng.dbg_pred_blocked(x0 + (x1 - x0) * i / n_, y0 + (y1 - y0) * i / n_) for i in range(n_ + 1))
    for k in range(120):
        a = rng.uniform(0, 2 * math.pi) if k else math.atan2(600 - ay, 800 - ax)
        gx, gy = ax + dg * math.cos(a), ay + dg * math.sin(a)
        if not (40 < gx < 1560 and 40 < gy < 1160) or not eng.dbg_free(gx - 6, gy - 6, 12): continue
        if LINE and not line_ok(gx, gy, ax, ay): continue
        b = math.radians(bear); ux, uy = math.cos(a), math.sin(a)   # u: sink -> guide (away from the sink)
        px, py = gx + dp * (ux * math.cos(b) - uy * math.sin(b)), gy + dp * (ux * math.sin(b) + uy * math.cos(b))
        if not (30 < px < 1570 and 30 < py < 1170) or eng.dbg_pred_blocked(px, py): continue
        found = (gx, gy, px, py); break
    if not found: return dict(base, skip='no_place')
    gx, gy, px, py = found
    cfg = dict(kw, sg_mode=kw.get('sg_mode', 1), no_spawn=1)
    eng.dbg_set_agent(guide, gx, gy, math.atan2(py - gy, px - gx), GE, float(sp), float(min(40, 2 * sp)), 800., 100., 400., 1.57, 1000.)
    for b_ in bys:
        for _ in range(40):
            a_ = rng.uniform(0, 2 * math.pi); d_ = rng.uniform(40, 160); bx, by = gx + d_ * math.cos(a_), gy + d_ * math.sin(a_)
            if eng.dbg_free(bx - 5, by - 5, 10): break
        else: return dict(base, skip='by_pos')
        eng.dbg_set_agent(b_, bx, by, rng.uniform(0, 2 * math.pi), 250., 10., 20., 600., 100., 400., 1.57, 1000.)
    eng.policy_init(nightsim.seed_key(seed), cfg)
    eng.dbg_freeze([guide] + bys)
    eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses(); eng.dbg_sg_sinks([pol_sink]); eng.dbg_sg_rects()
    for _ in range(2):
        eng.run_policy(1e9, eng.info()['time'] + 0.1); eng.dbg_true_poses()
    eng.dbg_freeze([])
    if not eng.dbg_add_predator(px, py, math.atan2(gy - py, gx - px), 200., False): return dict(base, skip='pred_pos')
    t0 = eng.info()['time']; died = None; dsink_death = None; last_g = None; dmin = 1e9; hist = []
    while eng.info()['time'] < t0 + T and eng.agents():
        eng.run_policy(t0 + T, eng.info()['time'] + (0.1 if TRACE else 0.5))
        ag = {a[0]: a for a in eng.agents()}
        if TRACE and guide in ag:
            g_ = ag[guide]; p_ = eng.predators()[0]; gs = eng.dbg_sg_guides()
            print(f"t{eng.info()['time']-t0:5.1f} g ({g_[1]:.0f},{g_[2]:.0f}) hd {g_[3]%6.283:.2f} e {g_[5]:.0f} | p ({p_[0]:.0f},{p_[1]:.0f}) hd {p_[2]%6.283:.2f} d {math.hypot(p_[0]-g_[1],p_[1]-g_[2]):.0f} mode {eng.dbg_pred_info()[0][3]} | dS {math.hypot(g_[1]-sx,g_[2]-sy):.0f} sg {gs}", flush=True)
        if guide in ag: last_g = (ag[guide][1], ag[guide][2])
        for kind, t, a, age, en in eng.pop_events():
            if a == guide and died is None and kind != 'fruit':
                died = (kind, round(t - t0, 1)); dsink_death = round(math.hypot(last_g[0] - sx, last_g[1] - sy), 1) if last_g else None
        p = eng.predators()[0]; dmin = min(dmin, math.hypot(p[0] - sx, p[1] - sy))
    t1 = eng.info()['time']; cnt, per = eng.dbg_sg_sinks()
    # hold phase: engine truth, did the predator stay?
    held = True; tend = t1 + HOLD
    while eng.info()['time'] < tend - 1e-6:
        if eng.agents(): eng.run_policy(tend, eng.info()['time'] + 1.0)
        else: eng.dbg_step(10)
        p = eng.predators()[0]; d = math.hypot(p[0] - sx, p[1] - sy); hist.append(round(d))
        if d > R: held = False
    p = eng.predators()[0]
    return dict(base, delivered=int(held), pred_end_d=round(math.hypot(p[0] - sx, p[1] - sy), 1), dmin_sink=round(dmin, 1),
                guide_died=died, dsink_death=dsink_death, guide_alive=int(any(a[0] == guide for a in eng.agents())),
                t_phase1=round(t1 - t0, 1), sg=cnt, by_alive=sum(1 for a in eng.agents() if a[0] in bys), dhist=hist[::30])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--sinks', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--dg', type=float, nargs='+', default=[120, 250]); ap.add_argument('--dp', type=float, nargs='+', default=[90, 140])
    ap.add_argument('--bear', type=float, nargs='+', default=[0, 90, 180]); ap.add_argument('--speeds', type=float, nargs='+', default=[10, 14])
    ap.add_argument('--minn', type=int, default=3); ap.add_argument('--minrob', type=float, default=0.); ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--T', type=float, default=120.); ap.add_argument('--hold', type=float, default=300.)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    for l in open(a.sinks):
        r = json.loads(l)
        if r.get('mode') == 'spec': SPECS[r['seed']] = [x for x in r['specs'] if x[5] and x[5][0][3] >= a.minrob and (len(x[5][0]) < 7 or x[5][0][6] >= MINKILL)]
        else: SINKS[r['seed']] = r['sinks']
    if SPECS: SINKS.update({k: [0] * len(v) for k, v in SPECS.items()})
    jobs = []
    for l, kw in cfgs.items():
        for s in parse_seeds(a.seeds):
            if s not in SINKS: continue
            ns = min(a.top, len(SPECS[s]) if SPECS else sum(1 for x in SINKS[s] if x[2] >= a.minn))
            for si in range(ns):
                for dg in a.dg:
                    for dp in a.dp:
                        for b in a.bear:
                            for sp in a.speeds: jobs.append((l, kw, s, si, dg, dp, b, sp, a.T, a.hold, a.minn))
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=2):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
