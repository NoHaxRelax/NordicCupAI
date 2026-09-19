"""Step-1 scenario test: guide ONE predator into a crevice (nightsim hooks; engine truth only to set up and grade).
usage: guide.py --configs CFG --seeds 1-32 --out rows.jsonl [--dg 150 300 500] [--dp 100 180 250] [--bear 0 90 180] [--speeds 10 13 16] [--T 60]
Setup per (seed, dg, dp, bear, speed): fresh map, keep 2 agents (bait, guide), give the policy true poses + true walls,
let it pick its best crevice (graded valid with engine truth), freeze the bait 9 units inside, put the guide dg units
out along the lane axis and the predator dp units from the guide at bearing `bear` (deg) from the lane axis, facing it.
Row: delivered (predator within 25 of the mouth for the last 3 s and bait alive), guide_alive, bait_alive, t_deliver,
guide energy used, min predator-bait distance, final guide state, skip reason."""
import argparse, json, math, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
from nightsim.trapsite import grade

def one(job):
    label, kw, seed, dg, dp, bear, sp, T = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    ags = eng.dbg_keep_agents(2)
    if len(ags) < 2: return dict(label=label, seed=seed, skip='agents')
    bait, guide = ags[0][0], ags[1][0]
    cfg = dict(kw, trap_mode=3, merge_anchored=1, pred_mode=1, pred_share=1, no_spawn=1, trap_bait_fixed=bait, trap_start=0)
    eng.policy_init(nightsim.seed_key(seed), cfg)
    eng.run_policy(1e9, eng.info()['time'] + 0.1)      # one policy call creates the minds
    eng.dbg_true_poses(); eng.dbg_load_walls()
    eng.run_policy(1e9, eng.info()['time'] + 0.1)      # sites computed
    sites = eng.dbg_sites()
    if not sites: return dict(label=label, seed=seed, skip='no_site')
    gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = sites[0]
    free, blocked, lane = grade(eng, gx, gy, ox, oy)
    if not (free and blocked and lane): return dict(label=label, seed=seed, skip='site_invalid')
    ax, ay = (mx - gx), (my - gy); n = math.hypot(ax, ay); ax, ay = ax / n, ay / n          # axis pointing OUT of the mouth
    def line_ok(x0, y0, x1, y1):
        n_ = max(2, int(math.hypot(x1 - x0, y1 - y0) / 8))
        return all(not eng.dbg_pred_blocked(x0 + (x1 - x0) * i / n_, y0 + (y1 - y0) * i / n_) for i in range(n_ + 1))
    # guide start: dg from the mouth within +-60 deg of the lane axis with a clear straight line to the lane point;
    # predator dp from the guide at bearing `bear` from that ray, with a clear line to the guide and a free spot
    found = None
    for phi in [0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60]:
        f = math.radians(phi); dx, dy = ax * math.cos(f) - ay * math.sin(f), ax * math.sin(f) + ay * math.cos(f)
        gsx, gsy = mx + dx * dg, my + dy * dg
        b = math.radians(bear); px, py = gsx + dp * (dx * math.cos(b) - dy * math.sin(b)), gsy + dp * (dx * math.sin(b) + dy * math.cos(b))
        if not eng.dbg_free(gsx - 5, gsy - 5, 10) or eng.dbg_pred_blocked(px, py): continue
        if line_ok(gsx, gsy, ox, oy) and line_ok(px, py, gsx, gsy): found = (gsx, gsy, px, py, phi); break
    if not found: return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='no_line')
    gsx, gsy, px, py, phi = found
    eng.dbg_set_agent(bait, gx, gy, math.atan2(-ay, -ax) + math.pi, 400., 10., 20., 500., 100., 400., 1.57, 1000.)
    eng.dbg_set_agent(guide, gsx, gsy, math.atan2(py - gsy, px - gsx), 300., float(sp), float(min(40, 2 * sp)), 800., 100., 400., 1.57, 1000.)
    if not eng.dbg_add_predator(px, py, math.atan2(gsy - py, gsx - px), 200., False): return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='pred_pos')
    # settle after teleporting: observations refresh and the policy's poses are re-synced twice (odometry/VO would drift)
    eng.dbg_freeze([bait, guide])
    for _ in range(3):
        eng.dbg_true_poses(); eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses(); eng.dbg_freeze([bait])
    t0 = eng.info()['time']; e0 = 300.; killed = {}; dmin = 1e9; near = 0.; t_del = None; states = []
    while eng.info()['time'] < t0 + T:
        eng.run_policy(t0 + T, eng.info()['time'] + 0.5)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'predator': killed[a] = round(t - t0, 1)
        pr = eng.predators()
        if pr:
            dmin = min(dmin, math.hypot(pr[0][0] - gx, pr[0][1] - gy))
            dm = math.hypot(pr[0][0] - mx, pr[0][1] - my)
            near = near + 0.5 if dm < 25 else 0.
            if near >= 3. and t_del is None and bait not in killed: t_del = round(eng.info()['time'] - t0, 1)
        r = eng.dbg_roles(); states.append(r[0][5] if r else -1)
        if bait in killed or guide in killed: break
    ag = {a[0]: a for a in eng.agents()}
    return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, delivered=int(t_del is not None and bait not in killed), t_deliver=t_del,
                guide_alive=int(guide in ag), bait_alive=int(bait in ag), used=round(e0 - ag[guide][5], 1) if guide in ag else None,
                dmin_bait=round(dmin, 1), states=''.join(str(x) for x in states[::4]), site=[round(gx), round(gy), round(ov), rear], phi=phi)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--dg', type=float, nargs='+', default=[150, 300, 500]); ap.add_argument('--dp', type=float, nargs='+', default=[100, 180, 250])
    ap.add_argument('--bear', type=float, nargs='+', default=[0, 90, 180]); ap.add_argument('--speeds', type=float, nargs='+', default=[10, 13, 16])
    ap.add_argument('--T', type=float, default=60.); ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, dg, dp, b, sp, a.T) for l, kw in cfgs.items() for s in parse_seeds(a.seeds) for dg in a.dg for dp in a.dp for b in a.bear for sp in a.speeds]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=2):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
