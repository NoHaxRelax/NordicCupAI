"""Late-game checkpoints with a working trap (nightsim; Linux, uses os.fork). For each seed: play a normal game with the
base config (map building on, no trap roles) to T0, then FORK once per variant so every variant continues from the
identical state. 'none' variants just continue; trap variants first set up a working trap at T0 (the policy picks its
crevice and a bait, the bait is teleported into the crevice, every predator is moved to the crevice mouth) and then
continue with the variant's policy to the horizon.
usage: lategame.py --base CFG.json:LABEL --variants VAR.json --seeds 1-96 --t0 900 1500 2100 --out rows.jsonl
VAR.json: {"label": {"setup": "none"|"hold_all", "params": {...}}}
Row: label, seed, t0, surv, score, fruit, alive0, preds0, placed, trees0, traj [[t, alive, preds, held, trees]], extra."""
import argparse, json, math, os, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
HORIZON = 3000.

def held_count(eng, mx, my):
    return sum(1 for p in eng.predators() if math.hypot(p[0] - mx, p[1] - my) < 35)

def run_to(eng, t_end, trap, sample=50.):
    traj = []; fe = 0.
    while eng.agents() and eng.info()['time'] < t_end - 1e-6:
        nxt = min(t_end, (math.floor(eng.info()['time'] / sample) + 1) * sample)
        eng.run_policy(t_end, nxt)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'fruit': fe += en
        tr = eng.dbg_trap() if trap else None
        mx, my = (tr[0], tr[1]) if tr else (trap or (None, None))
        traj.append([round(eng.info()['time']), len(eng.agents()), len(eng.predators()),
                     held_count(eng, mx, my) if mx is not None else 0, len(eng.trees())])
    return traj, fe

def variant(eng, label, spec, seed, t0, snap):
    setup = spec.get('setup', 'none'); params = spec.get('params', {}); extra = {}
    placed = 0; trap_pt = None
    if setup == 'poses':   # control: only the pose reset the trap setup uses
        eng.dbg_true_poses(); eng.dbg_set_params(params)
    elif setup != 'none':
        # 1) choose the crevice and a bait with an immediate replacement (no keeper), 2) teleport it in, 3) full params
        if spec.get('true_walls'): eng.dbg_load_walls()   # idealised 'working trap': the policy gets the true wall map
        eng.dbg_set_params(dict(params, trap_mode=1))   # site constraints of the variant first: sites are recomputed every 2 s
        eng.run_policy(HORIZON, eng.info()['time'] + 2.2)
        eng.dbg_set_params(dict(params, trap_mode=max(2, params.get('trap_mode', 2)), keeper_mode=0, trap_start=0, bait_on_sight=0))
        tr = None; rep = -1
        for _ in range(20):
            eng.run_policy(HORIZON, eng.info()['time'] + 0.1)
            tr = eng.dbg_trap(); roles = eng.dbg_roles()
            cand = [r for r in roles if r[1] and (r[3] >= 0 or r[2] >= 0)]
            if tr and cand:
                rep = cand[0][3] if cand[0][3] >= 0 else cand[0][2]; break
        if not tr or rep < 0: return dict(label=label, seed=seed, t0=t0, skip='no_trap', **snap)
        mx, my, gx, gy = tr[0], tr[1], tr[2], tr[3]
        if spec.get('true_walls'):
            from nightsim.trapsite import grade
            ox_, oy_ = mx + (mx - gx) / max(1e-6, math.hypot(mx - gx, my - gy)) * 60., my + (my - gy) / max(1e-6, math.hypot(mx - gx, my - gy)) * 60.
            f_, b_, l_ = grade(eng, gx, gy, ox_, oy_)
            if not (f_ and b_ and l_): return dict(label=label, seed=seed, t0=t0, skip='site_invalid', **snap)
        ax, ay = mx - gx, my - gy; n_ = math.hypot(ax, ay) or 1.; ax, ay = ax / n_, ay / n_
        eng.dbg_teleport(rep, gx, gy, math.atan2(ay, ax))
        if spec.get('bait_e'): eng.dbg_set_energy(rep, float(spec['bait_e']))
        extra['bait_id'] = rep
        eng.dbg_true_poses(); eng.run_policy(HORIZON, eng.info()['time'] + 0.1); eng.dbg_true_poses()
        eng.dbg_set_params(dict(params, trap_start=0))
        # every predator to the mouth, rows of five across the lane
        prs = eng.predators() if setup == 'hold_all' else []
        for i in range(len(prs)):
            k = placed; ok = False
            for tries in range(12):
                back = 12. + 8. * ((k + tries) // 5); off = (((k + tries) % 5) - 2) * 4.0
                hx, hy = mx + ax * back - ay * off, my + ay * back + ax * off
                if eng.dbg_move_predator(i, hx, hy, math.atan2(gy - hy, gx - hx), 200.): ok = True; break
            placed += ok
        trap_pt = (mx, my)
        extra['held0'] = held_count(eng, mx, my)
    else:
        eng.dbg_set_params(params)
    eng.dbg_deaths()   # drop records from before the fork point
    if spec.get('probe'):   # 10-s samples for 300 s: held predators and whether the first bait is alive
        probe = []
        for _ in range(30):
            if not eng.agents(): break
            eng.run_policy(HORIZON, eng.info()['time'] + 10.)
            ids = {a[0] for a in eng.agents()}
            probe.append([held_count(eng, trap_pt[0], trap_pt[1]) if trap_pt else 0, int(extra.get('bait_id', -1) in ids), len(ids)])
        extra['probe'] = probe
    traj, fe = run_to(eng, HORIZON, trap_pt if setup not in ('none', 'poses') else None)
    D = eng.dbg_deaths()   # (cause, t, age, e, maxe, speed, sprint, x, y, npred150, dpred, prest, nearwall, pop, npred, evading, old, haspost)
    tr_ = eng.dbg_trap()
    mx_, my_ = (tr_[0], tr_[1]) if tr_ else (trap_pt or (None, None))
    def near(d): return mx_ is not None and math.hypot(d[7] - mx_, d[8] - my_) < 120
    extra['kills'] = sum(1 for d in D if d[0] == 1); extra['starved'] = sum(1 for d in D if d[0] == 0)
    extra['kills_near_trap'] = sum(1 for d in D if d[0] == 1 and near(d)); extra['starved_near_trap'] = sum(1 for d in D if d[0] == 0 and near(d))
    extra['kills_first100'] = sum(1 for d in D if d[0] == 1 and d[1] < t0 + 100); extra['starved_first100'] = sum(1 for d in D if d[0] == 0 and d[1] < t0 + 100)
    info = eng.info()
    if setup not in ('none', 'poses'):
        tr = eng.dbg_trap()
        if tr: extra['funnel'] = list(tr[8]) if len(tr) > 8 else None
    return dict(label=label, seed=seed, t0=t0, surv=round(info['time'], 1), score=round(info['score'], 3), fruit_after=round(fe / 1000, 3),
                placed=placed, traj=traj, **snap, **extra)

def one(job):
    seed, t0, base, variants = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([]); eng.pop_events()
    eng.policy_init(nightsim.seed_key(seed), dict(base, trap_mode=1))
    while eng.agents() and eng.info()['time'] < t0 - 1e-6:
        eng.run_policy(t0, min(t0, eng.info()['time'] + 100.)); eng.pop_events()
    if not eng.agents(): return [dict(label=l, seed=seed, t0=t0, skip='dead_before_t0') for l in variants]
    snap = dict(alive0=len(eng.agents()), preds0=len(eng.predators()), trees0=len(eng.trees()), score0=round(eng.info()['score'], 3))
    out = []
    for label, spec in variants.items():
        r, w = os.pipe(); pid = os.fork()
        if pid == 0:
            os.close(r)
            try: res = variant(eng, label, spec, seed, t0, snap)
            except Exception as e: res = dict(label=label, seed=seed, t0=t0, skip='error', err=repr(e)[:200])
            os.write(w, json.dumps(res).encode()); os.close(w); os._exit(0)
        os.close(w); buf = b''
        while True:
            chunk = os.read(r, 65536)
            if not chunk: break
            buf += chunk
        os.close(r); os.waitpid(pid, 0)
        out.append(json.loads(buf) if buf else dict(label=label, seed=seed, t0=t0, skip='child_failed'))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--variants', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--t0', type=float, nargs='+', default=[900., 1500., 2100.]); ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bf, bl = a.base.split(':'); base = json.load(open(bf))[bl]
    variants = json.load(open(a.variants))
    jobs = [(s, t0, base, variants) for t0 in a.t0 for s in parse_seeds(a.seeds)]
    t0_ = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for rows in pool.imap_unordered(one, jobs, chunksize=1):
            for r in rows: f.write(json.dumps(r) + '\n'); n += 1
            f.flush()
    print(f'done {n} rows in {time.time()-t0_:.0f}s', flush=True)
