"""Holding scenario (nightsim hooks; engine truth only to set up and grade): a small colony forages around a crevice
trap; once the policy has a bait in the crevice, one awake predator is placed just outside the mouth. Run T seconds
and measure how long the predator stays held and why the hold ends.
usage: hold.py --configs CFG.json --seeds 1-64 --out rows.jsonl [--n 8] [--r 250] [--T 600]
Row: label, seed, held_frac (share of 1-s samples with the predator within 35 of the mouth), t_release (first time it
was away >= 10 s), bait_changes, bait_deaths (starved in the crevice), kills (colony agents eaten), starved,
alive_end, rotated, rear_ok."""
import argparse, json, math, sys, time, pathlib, random
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
from nightsim.trapsite import grade

def one(job):
    label, kw, seed, N, R, T = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    ags = eng.dbg_keep_agents(N)
    if len(ags) < N: return dict(label=label, seed=seed, skip='agents')
    cfg = dict(kw, trap_mode=2, merge_anchored=1, pred_mode=1, trap_start=0)
    eng.policy_init(nightsim.seed_key(seed), cfg)
    eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses(); eng.dbg_load_walls()
    eng.run_policy(1e9, eng.info()['time'] + 0.1)
    sites = eng.dbg_sites()
    if not sites: return dict(label=label, seed=seed, skip='no_site')
    gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = sites[0]
    free, blocked, lane = grade(eng, gx, gy, ox, oy)
    if not (free and blocked and lane): return dict(label=label, seed=seed, skip='site_invalid')
    ax, ay = (mx - gx), (my - gy); n_ = math.hypot(ax, ay); ax, ay = ax / n_, ay / n_   # out of the mouth
    rng = random.Random(seed)
    for a in ags:   # the colony: spread 60..R around the trap, not in the lane in front of the mouth
        for _ in range(60):
            ang = rng.uniform(0, 2 * math.pi); d = rng.uniform(60, R); x, y = gx + d * math.cos(ang), gy + d * math.sin(ang)
            ahead = (x - mx) * ax + (y - my) * ay; lat = abs(-(x - mx) * ay + (y - my) * ax)
            if ahead > 0 and lat < 40: continue
            if eng.dbg_free(x - 5, y - 5, 10): break
        else: return dict(label=label, seed=seed, skip='place')
        eng.dbg_set_agent(a[0], x, y, rng.uniform(0, 2 * math.pi), rng.uniform(250, 400), 10., 20., 600., 100., 400., 1.57, 1000.)
    for _ in range(3):
        eng.dbg_true_poses(); eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses()
    # let the policy put a bait into the crevice (no predator yet)
    t0 = eng.info()['time']; bait = -1
    while eng.info()['time'] < t0 + 90:
        eng.run_policy(1e9, eng.info()['time'] + 1.0); eng.pop_events()
        r = eng.dbg_roles(); ag = {a[0]: a for a in eng.agents()}
        if r and r[0][2] >= 0 and r[0][2] in ag and math.hypot(ag[r[0][2]][1] - gx, ag[r[0][2]][2] - gy) < 5: bait = r[0][2]; break
    if bait < 0: return dict(label=label, seed=seed, skip='no_bait', rear_ok=rear)
    px, py = mx + ax * 14., my + ay * 14.
    if not eng.dbg_add_predator(px, py, math.atan2(gy - py, gx - px), 200., False): return dict(label=label, seed=seed, skip='pred_pos')
    t1 = eng.info()['time']; held = n = 0; away = 0; t_rel = None; kills = starved = 0; bait_deaths = 0; changes = 0; last_bait = bait
    while eng.info()['time'] < t1 + T and eng.agents():
        eng.run_policy(t1 + T, eng.info()['time'] + 1.0)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'predator': kills += 1
            elif kind == 'starvation':
                starved += 1
                if a == last_bait: bait_deaths += 1
        pr = eng.predators(); n += 1
        h = bool(pr) and math.hypot(pr[0][0] - mx, pr[0][1] - my) < 35
        held += h; away = 0 if h else away + 1
        if away >= 10 and t_rel is None: t_rel = round(eng.info()['time'] - t1 - 10)
        r = eng.dbg_roles()
        if r and r[0][2] >= 0 and r[0][2] != last_bait: changes += 1; last_bait = r[0][2]
    info = eng.dbg_trap()
    return dict(label=label, seed=seed, held_frac=round(held / max(1, n), 3), t_release=t_rel, bait_changes=changes, bait_deaths=bait_deaths,
                kills=kills, starved=starved, alive_end=len(eng.agents()), rear_ok=rear, T=round(eng.info()['time'] - t1))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--r', type=float, default=250.); ap.add_argument('--T', type=float, default=600.)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, a.n, a.r, a.T) for l, kw in cfgs.items() for s in parse_seeds(a.seeds)]
    t0 = time.time(); k = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            if r: f.write(json.dumps(r) + '\n'); f.flush(); k += 1
    print(f'done {k} in {time.time()-t0:.0f}s', flush=True)
