"""Refuge scenario test (nightsim hooks; engine truth only to set up and grade): ONE agent near a narrow gap, ONE fresh
predator behind it. Does the agent get into the gap alive, and does the predator then stay at the mouth?
usage: refuge.py --configs CFG.json --seeds 1-48 --out rows.jsonl [--da 30 60 90] [--dp 50 80 110] [--speeds 10 13 16] [--T 40]
Row: label, seed, da (agent distance from the mouth), dp (predator distance behind the agent), speed, killed, t_kill,
in_gap (agent reached the goal), held (predator ticks within 45 of the mouth after the agent got in), fdist."""
import argparse, json, math, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
from nightsim.trapsite import grade

def one(job):
    label, kw, seed, da, dp, sp, T = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    ags = eng.dbg_keep_agents(1)
    if len(ags) < 1: return dict(label=label, seed=seed, skip='agents')
    aid = ags[0][0]
    cfg = dict(kw, trap_mode=1, merge_anchored=1, pred_mode=1, no_spawn=1)   # trap_mode 1 = walls + sites only (no bait/guide)
    eng.policy_init(nightsim.seed_key(seed), cfg)
    eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses(); eng.dbg_load_walls()
    eng.run_policy(1e9, eng.info()['time'] + 0.1)
    sites = eng.dbg_sites()
    if not sites: return dict(label=label, seed=seed, skip='no_site')
    gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = sites[0]
    free, blocked, lane = grade(eng, gx, gy, ox, oy)
    if not (free and blocked and lane): return dict(label=label, seed=seed, skip='site_invalid')
    ax, ay = (mx - gx), (my - gy); n = math.hypot(ax, ay); ax, ay = ax / n, ay / n          # axis pointing OUT of the mouth
    # agent da outside the mouth on the lane axis, facing the mouth; predator dp further out, facing the agent
    asx, asy = mx + ax * da, my + ay * da
    px, py = mx + ax * (da + dp), my + ay * (da + dp)
    if not eng.dbg_free(asx - 5, asy - 5, 10) or eng.dbg_pred_blocked(px, py): return dict(label=label, seed=seed, da=da, dp=dp, speed=sp, skip='blocked')
    eng.dbg_set_agent(aid, asx, asy, math.atan2(-ay, -ax), 150., float(sp), float(min(40, 2 * sp)), 800., 100., 400., 1.57, 1000.)
    eng.dbg_freeze([aid])
    for _ in range(3):
        eng.dbg_true_poses(); eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses(); eng.dbg_freeze([])
    # the predator is added only after the warm-up (a fresh predator moves 15 per tick and would reach a frozen agent)
    if not eng.dbg_add_predator(px, py, math.atan2(-ay, -ax), 200., False): return dict(label=label, seed=seed, da=da, dp=dp, speed=sp, skip='pred')
    killed = 0; tk = None; in_gap = 0; held = 0; t0 = eng.info()['time']
    while eng.info()['time'] < t0 + T:
        eng.run_policy(t0 + T, eng.info()['time'] + 0.2)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'predator' and a == aid: killed = 1; tk = round(t - t0, 1)
        ag = [a for a in eng.agents() if a[0] == aid]; pr = eng.predators()
        if ag and math.hypot(ag[0][1] - gx, ag[0][2] - gy) < 4.: in_gap = 1
        if in_gap and pr and math.hypot(pr[0][0] - mx, pr[0][1] - my) < 45.: held += 1
        if killed or not eng.agents(): break
    ag = [a for a in eng.agents() if a[0] == aid]; pr = eng.predators()
    fd = math.hypot(ag[0][1] - pr[0][0], ag[0][2] - pr[0][1]) if ag and pr else None
    info = eng.dbg_eval()
    return dict(label=label, seed=seed, da=da, dp=dp, speed=sp, killed=killed, t_kill=tk, in_gap=in_gap, held=held,
                fdist=fd and round(fd), ev=info.get('refuge_events', 0), holds=info.get('refuge_holds', 0))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--da', type=float, nargs='+', default=[30, 60, 90]); ap.add_argument('--dp', type=float, nargs='+', default=[50, 80, 110])
    ap.add_argument('--speeds', type=float, nargs='+', default=[10, 13, 16]); ap.add_argument('--T', type=float, default=40.)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, da, dp, sp, a.T) for l, kw in cfgs.items() for s in parse_seeds(a.seeds) for da in a.da for dp in a.dp for sp in a.speeds]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=4):
            if r: f.write(json.dumps(r) + '\n'); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
