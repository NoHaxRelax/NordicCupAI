"""Trap-site tests (nightsim; engine truth used ONLY to grade). usage: trapsite.py --configs CFG --seeds 1-32 --out rows.jsonl
Stage A: run the policy (trap_mode=1) for --t seconds without predators, read its crevice sites, grade the top ones:
  goal free for an agent; every point within 15 of the goal is predator-blocked; the approach point 60 out is predator-free.
Stage B (hold): fresh game, one agent frozen at the graded goal, one awake predator at the approach point facing it;
  run --hold seconds; record whether the bait survives and the predator's closest approach."""
import argparse, json, math, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds

def grade(eng, gx, gy, ox, oy):
    free = eng.dbg_free(gx - 3, gy - 3, 6)
    blocked = all(eng.dbg_pred_blocked(gx + r * math.cos(a), gy + r * math.sin(a))
                  for r in (0., 5., 10., 14.9) for a in [k * math.pi / 18 for k in range(36)])
    lane = not eng.dbg_pred_blocked(ox, oy)
    return free, blocked, lane

def hold(seed, kw, gx, gy, ox, oy, T):
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    aid, x, y = eng.dbg_keep_agent(-1)
    eng.dbg_set_agent(aid, gx, gy, 0., 400., 10., 20., 500., 100., 400., 1.57, 1000.)
    ang = math.atan2(gy - oy, gx - ox)
    if not eng.dbg_add_predator(ox, oy, ang, 200., False): return dict(hold='pred_not_free')
    eng.policy_init(nightsim.seed_key(seed), dict(kw, test_freeze=1, no_spawn=1, pred_mode=0))
    t0 = eng.info()['time']; dmin = 1e9; killed = 0
    while eng.info()['time'] < t0 + T:
        eng.run_policy(t0 + T, eng.info()['time'] + 0.2)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'predator' and a == aid: killed = 1
        pr = eng.predators()
        if pr: dmin = min(dmin, math.hypot(pr[0][0] - gx, pr[0][1] - gy))
        if killed: break
    return dict(hold_killed=killed, hold_dmin=round(dmin, 2), hold_t=round(eng.info()['time'] - t0, 1))

def one(job):
    label, kw, seed, T, TH, top = job
    import nightsim
    t0 = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    eng.policy_init(nightsim.seed_key(seed), dict(kw, trap_mode=1))
    eng.run_policy(T, T)
    sites = eng.dbg_sites()
    out = dict(label=label, seed=seed, n_sites=len(sites), walls=sites[0][10] if sites else None, alive=len(eng.agents()), graded=[])
    seen = set()
    for st in sites:
        gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = st
        key = (round(gx), round(gy))
        if key in seen: continue
        seen.add(key)
        free, blocked, lane = grade(eng, gx, gy, ox, oy)
        g = dict(goal=[round(gx, 1), round(gy, 1)], overlap=round(ov, 1), gap=round(gap, 1), rear=rear, free=free, blocked=blocked, lane=lane)
        if free and blocked and lane and TH > 0: g.update(hold(seed, kw, gx, gy, ox, oy, TH))
        out['graded'].append(g)
        if len(out['graded']) >= top: break
    out['wall_s'] = round(time.perf_counter() - t0, 1)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--t', type=float, default=300.); ap.add_argument('--hold', type=float, default=60.); ap.add_argument('--top', type=int, default=3)
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, a.t, a.hold, a.top) for l, kw in cfgs.items() for s in parse_seeds(a.seeds)]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
