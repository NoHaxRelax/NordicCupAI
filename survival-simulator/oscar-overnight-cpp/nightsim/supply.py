"""Can the colony keep a crevice trap supplied late in the game? (nightsim; Linux, os.fork; engine truth for setup/oracles)
For each seed: play the base config (map building on, no trap roles) to T0, then FORK once per variant so every variant
continues from the identical state. Trap variants set up a working trap at T0 exactly like lategame.py hold_all (true
walls, policy-picked crevice, bait teleported to the goal, every predator moved to the mouth) and then differ only in
HOW the bait is kept supplied and whether predators that spawn later are also held:
  supply 'policy'  the policy's own replacement logic (walks a colony member in through the rear entrance)
  supply 'free'    oracle: the bait never dies (max_age and energy reset every second) - zero colony cost
  supply 'young'   oracle: ~3 s before the bait would die, the YOUNGEST colony agent is teleported into the goal and
                   frozen there (costs the colony one agent per bait, no travel risk) - 'can the colony afford it'
  supply 'fed'     as 'young', but the newborn is first fuelled to its max energy, paid for by the richest colony
                   members (never below 150) - the 'reserve fruit to fuel young baits' idea, logistics-free
  hold_new 1       oracle guides: every predator that is not at the mouth is moved there each second
usage: supply.py --base CFG.json:LABEL --variants VAR.json --seeds 1-240 --t0 1500 1900 2300 --out rows.jsonl
VAR.json: {"label": {"setup": "none"|"hold_all", "supply": "policy|free|young|fed", "hold_new": 0|1, "params": {...}}}"""
import argparse, json, math, os, sys, time, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
HORIZON = 3000.
DT = 0.1

def held_count(eng, mx, my):
    return sum(1 for p in eng.predators() if math.hypot(p[0] - mx, p[1] - my) < 50)

def time_to_death(a, cap=6.):   # seconds until the engine would starve agent tuple a (capped)
    """engine order per tick: age += dt; energy -= dt*1.0 (every biome drains 1/s); dead if <= 0; senescence after max_age"""
    age, e, ma = a[4], a[5], a[13]; t = 0.
    while t < cap:
        age += DT; e -= DT
        if e <= 0: return t
        if age > ma: e -= 0.01 * age
        t += DT
    return cap

def place_pred(eng, i, mx, my, ax, ay, gx, gy, k0):
    for tries in range(40):
        k = k0 + tries
        back = 12. + 8. * (k // 5); off = ((k % 5) - 2) * 4.0
        hx, hy = mx + ax * back - ay * off, my + ay * back + ax * off
        if eng.dbg_move_predator(i, hx, hy, math.atan2(gy - hy, gx - hx), 200.): return True
    return False

def setup_trap(eng, params, spec):
    """lategame.py hold_all: policy picks crevice + bait; bait teleported in; all predators to the mouth. -> dict or skip reason"""
    if spec.get('anchor'): eng.dbg_true_poses()   # early checkpoints: anchor every group (true poses) so a site can be picked
    if spec.get('true_walls', 1): eng.dbg_load_walls()
    eng.dbg_set_params(dict(params, trap_mode=1))
    eng.run_policy(HORIZON, eng.info()['time'] + 2.2)
    eng.dbg_set_params(dict(params, trap_mode=max(2, params.get('trap_mode', 2)), keeper_mode=0, trap_start=0, bait_on_sight=0))
    tr = None; rep = -1
    for _ in range(20):
        eng.run_policy(HORIZON, eng.info()['time'] + 0.1)
        tr = eng.dbg_trap(); roles = eng.dbg_roles()
        cand = [r for r in roles if r[1] and (r[3] >= 0 or r[2] >= 0)]
        if tr and cand:
            rep = cand[0][3] if cand[0][3] >= 0 else cand[0][2]; break
    if not tr or rep < 0: return 'no_trap'
    mx, my, gx, gy = tr[0], tr[1], tr[2], tr[3]
    ax, ay = mx - gx, my - gy; n_ = math.hypot(ax, ay) or 1.; ax, ay = ax / n_, ay / n_
    if spec.get('true_walls', 1):
        from nightsim.trapsite import grade
        f_, b_, l_ = grade(eng, gx, gy, mx + ax * 60., my + ay * 60.)
        if not (f_ and b_): return 'site_invalid_' + ('f' if not f_ else '') + ('b' if not b_ else '')
    if spec.get('supply') in ('young', 'fed'):   # oracle supply starts with the youngest agent too
        ags = [a for a in eng.agents()]
        if ags: rep = min(ags, key=lambda a: a[4])[0]
    eng.dbg_teleport(rep, gx, gy, math.atan2(ay, ax))
    eng.dbg_true_poses(); eng.run_policy(HORIZON, eng.info()['time'] + 0.1); eng.dbg_true_poses()
    eng.dbg_set_params(dict(params, trap_start=0))
    placed = 0
    for i in range(len(eng.predators())): placed += place_pred(eng, i, mx, my, ax, ay, gx, gy, placed)
    if placed == 0 and len(eng.predators()) > 0: return 'no_place'
    return dict(mx=mx, my=my, gx=gx, gy=gy, ax=ax, ay=ay, bait=rep, placed=placed, lane=int(bool(l_)) if spec.get('true_walls', 1) else None)

def fuel(eng, aid, floor=150.):
    """top aid up to its max energy, paid by the richest other agents (never below floor). -> energy actually added"""
    A = {a[0]: a for a in eng.agents()}
    if aid not in A: return 0.
    need = A[aid][6] - A[aid][5]; paid = 0.
    for a in sorted((a for a in A.values() if a[0] != aid), key=lambda a: -a[5]):
        if paid >= need: break
        take = min(need - paid, a[5] - floor)
        if take <= 0: break
        eng.dbg_set_energy(a[0], a[5] - take); paid += take
    if paid > 0: eng.dbg_set_energy(aid, A[aid][5] + paid)
    return paid

def continue_trap(eng, spec, T, t_end):
    """1-s control loop with the supply oracle + optional predator holding. -> (probe, traj, sup)"""
    supply = spec.get('supply', 'policy'); hold_new = spec.get('hold_new', 0)
    mx, my, gx, gy, ax, ay = T['mx'], T['my'], T['gx'], T['gy'], T['ax'], T['ay']
    face = math.atan2(ay, ax)
    cur = T['bait']; baits = [cur]
    sup = dict(n=0, ages=[], fed=0., dry=0, moved=0, secs=0, bait_s=0, moved_s=0, notrap_s=0, retired_max=0, roles_max=0)
    if supply != 'policy':   # oracle supply: put the first bait back on the hold point (the policy walks it ~10 deeper in its first tick)
        eng.dbg_teleport(cur, gx, gy, face); eng.dbg_freeze(baits); eng.dbg_set_params({'trap_bait_fixed': cur})
    t_start = eng.info()['time']; probe = []; traj = []; nid0 = eng.info()['next_agent_id']
    next_traj = (math.floor(t_start / 50.) + 1) * 50.
    while eng.agents() and eng.info()['time'] < t_end - 1e-6:
        eng.run_policy(t_end, min(t_end, eng.info()['time'] + 1.0)); eng.pop_events()
        now = eng.info()['time']
        A = {a[0]: a for a in eng.agents()}
        if not A: break
        active = now - t_start < spec.get('hold_T', 1e9)   # optional limited trap: after hold_T no supply, no holding
        if not active:
            supply_now = 'none'
            if supply == 'free' and not sup.get('ended') and cur in A:   # the immortal bait ages normally from now on
                a = A[cur]; eng.dbg_set_agent(cur, a[1], a[2], a[3], a[5], a[7], a[8], a[6], a[9], a[10], a[12], a[4])
            sup['ended'] = 1
        else: supply_now = supply
        if supply_now == 'free':
            if cur not in A:   # died anyway (eaten): next one from the colony, then immortal
                cur = min(A.values(), key=lambda a: a[4])[0]; eng.dbg_teleport(cur, gx, gy, face)
                baits.append(cur); sup['n'] += 1; sup['ages'].append(round(A[cur][4], 1))
                eng.dbg_freeze([b for b in baits if b in A]); eng.dbg_set_params({'trap_bait_fixed': cur})
            a = A[cur]
            eng.dbg_set_agent(cur, a[1], a[2], a[3], a[6], a[7], a[8], a[6], a[9], a[10], a[12], 1e7)
        elif supply_now in ('young', 'fed', 'old'):
            if cur not in A or time_to_death(A[cur]) < 3.:
                cands = [a for a in A.values() if a[0] not in baits]
                if len(cands) < spec.get('min_colony', 1): cands = []   # smarter supply: never take one of the last few agents
                if cands:
                    if supply_now == 'old':   # the cheapest sacrifice: the agent closest to death that can still hold >= 30 s
                        life = {a[0]: time_to_death(a, 400.) for a in cands}
                        ok = [a for a in cands if life[a[0]] >= 30.]
                        nb = min(ok, key=lambda a: life[a[0]]) if ok else max(cands, key=lambda a: life[a[0]])
                    else: nb = min(cands, key=lambda a: a[4])
                    eng.dbg_teleport(nb[0], gx, gy, face)
                    if supply_now == 'fed': sup['fed'] += fuel(eng, nb[0])
                    cur = nb[0]; baits.append(cur); sup['n'] += 1; sup['ages'].append(round(nb[4], 1))
                    eng.dbg_freeze([b for b in baits if b in A]); eng.dbg_set_params({'trap_bait_fixed': cur})
                else: sup['dry'] += 1
        bait_here = any(math.hypot(a[1] - gx, a[2] - gy) < 6. for a in eng.agents())
        if hold_new and active and (bait_here or not spec.get('hold_gate', 1)):   # oracle guides only while a bait holds the crevice
            prs = eng.predators(); k = held_count(eng, mx, my)
            for i, p in enumerate(prs):
                if math.hypot(p[0] - mx, p[1] - my) >= 50:
                    if place_pred(eng, i, mx, my, ax, ay, gx, gy, k): k += 1; sup['moved'] += 1
        if 'colony_end' not in sup and not any(x not in baits for x in A): sup['colony_end'] = round(now, 1)   # only baits left
        # diagnostics: is a live agent holding the goal, is the policy's trap still where the predators are held
        sup['secs'] += 1
        if any(math.hypot(a[1] - gx, a[2] - gy) < 6. for a in A.values()): sup['bait_s'] += 1
        tr_ = eng.dbg_trap()
        if not tr_: sup['notrap_s'] += 1
        else:
            if math.hypot(tr_[0] - mx, tr_[1] - my) > 10.: sup['moved_s'] += 1
            sup['retired_max'] = max(sup['retired_max'], tr_[5])
        rel = now - t_start
        if rel <= 300.5 and abs(rel - round(rel / 10.) * 10.) < 0.51:
            probe.append([round(rel), held_count(eng, mx, my), len(eng.predators()), len(A)])
        if now >= next_traj - 1e-6:
            traj.append([round(now), len(A), len(eng.predators()), held_count(eng, mx, my)]); next_traj += 50.
    sup['births'] = eng.info()['next_agent_id'] - nid0
    if hasattr(eng, 'dbg_nest'):
        nz = eng.dbg_nest()
        if nz: sup['nest'] = dict(cadets=nz[2], sent=nz[3], arrived=nz[4], timeouts=nz[5], sent_e=round(nz[6] / max(1, nz[3]), 1), sent_age=round(nz[7] / max(1, nz[3]), 1))
    return probe, traj, sup

def continue_plain(eng, t_end):
    t_start = eng.info()['time']; traj = []; nid0 = eng.info()['next_agent_id']
    while eng.agents() and eng.info()['time'] < t_end - 1e-6:
        nxt = min(t_end, (math.floor(eng.info()['time'] / 50.) + 1) * 50.)
        eng.run_policy(t_end, nxt); eng.pop_events()
        traj.append([round(eng.info()['time']), len(eng.agents()), len(eng.predators()), 0])
    return traj, dict(births=eng.info()['next_agent_id'] - nid0)

def variant(eng, label, spec, seed, t0, snap):
    setup = spec.get('setup', 'none'); params = spec.get('params', {})
    eng.dbg_deaths()
    if setup == 'none':
        eng.dbg_set_params(params)
        traj, sup = continue_plain(eng, HORIZON); probe = None; T = {}
    else:
        T = setup_trap(eng, params, spec)
        if isinstance(T, str): return dict(label=label, seed=seed, t0=t0, skip=T, **snap)
        eng.dbg_deaths()
        probe, traj, sup = continue_trap(eng, spec, T, HORIZON)
    D = eng.dbg_deaths()   # (cause, t, age, e, ...)
    info = eng.info()
    ex = dict(kills=sum(1 for d in D if d[0] == 1), starved=sum(1 for d in D if d[0] == 0),
              kills_50=sum(1 for d in D if d[0] == 1 and d[1] < t0 + 50), starved_50=sum(1 for d in D if d[0] == 0 and d[1] < t0 + 50),
              starved_50_age=[round(d[2]) for d in D if d[0] == 0 and d[1] < t0 + 50][:40],
              kills_300=sum(1 for d in D if d[0] == 1 and d[1] < t0 + 300), starved_300=sum(1 for d in D if d[0] == 0 and d[1] < t0 + 300))
    return dict(label=label, seed=seed, t0=t0, surv=round(info['time'], 1), score=round(info['score'], 3), probe=probe, traj=traj,
                sup=sup, placed=T.get('placed') if isinstance(T, dict) else None, lane=T.get('lane') if isinstance(T, dict) else None, **snap, **ex)

def one(job):
    seed, t0s, base, variants = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([]); eng.pop_events()
    eng.policy_init(nightsim.seed_key(seed), dict(base, trap_mode=1))
    out = []; nid_prev = None; t_prev = None
    for t0 in sorted(t0s):
        while eng.agents() and eng.info()['time'] < t0 - 1e-6:
            if eng.info()['time'] >= t0 - 100. - 1e-6 and nid_prev is None: nid_prev, t_prev = eng.info()['next_agent_id'], eng.info()['time']
            eng.run_policy(t0, min(t0, eng.info()['time'] + 50.)); eng.pop_events()
        if not eng.agents():
            out.extend(dict(label=l, seed=seed, t0=t0, skip='dead_before_t0') for l in variants); continue
        A = eng.agents(); ages = sorted(a[4] for a in A); en = [a[5] for a in A]
        snap = dict(alive0=len(A), preds0=len(eng.predators()), score0=round(eng.info()['score'], 3),
                    births_prev100=(eng.info()['next_agent_id'] - nid_prev) if nid_prev is not None else None,
                    young30=sum(1 for x in ages if x < 30), young60=sum(1 for x in ages if x < 60),
                    e_mean=round(sum(en) / len(en), 1), e_sum=round(sum(en)), rich=sum(1 for x in en if x > 300))
        for label, spec in variants.items():
            r, w = os.pipe(); pid = os.fork()
            if pid == 0:
                os.close(r)
                try: res = variant(eng, label, spec, seed, t0, snap)
                except Exception as e: res = dict(label=label, seed=seed, t0=t0, skip='error', err=repr(e)[:300])
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
    ap.add_argument('--t0', type=float, nargs='+', default=[1500., 1900., 2300.]); ap.add_argument('--workers', type=int, default=22)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bf, bl = a.base.split(':'); base = json.load(open(bf))[bl]
    variants = json.load(open(a.variants))
    jobs = [(s, [t0], base, variants) for t0 in sorted(a.t0) for s in parse_seeds(a.seeds)]   # earliest checkpoints first
    t0_ = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for rows in pool.imap_unordered(one, jobs, chunksize=1):
            for r in rows: f.write(json.dumps(r) + '\n'); n += 1
            f.flush()
    print(f'done {n} rows in {time.time()-t0_:.0f}s', flush=True)
