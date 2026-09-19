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

HELD = int(__import__('os').environ.get('NIGHT_HELD', '0'))
RELAY = int(__import__('os').environ.get('NIGHT_RELAY', '0'))
NOLINE = int(__import__('os').environ.get('NIGHT_NOLINE', '0'))   # accept placements whose straight lane is blocked (routing test)
_E = __import__('os').environ
GE = float(_E.get('NIGHT_GE', '300'))        # guide start energy (max 800 -> sprint cap at 160)
TRACE = int(_E.get('NIGHT_TRACE', '0'))      # print one line per tick (use with a single job, --workers 1)
NPRED = int(_E.get('NIGHT_NPRED', '1'))      # edge case 2: predators chasing (extra ones placed 60-140 from the guide)
NBY = int(_E.get('NIGHT_NBY', '0'))          # edge case 3: extra free agents (bystanders) 40-160 from the guide   # place a third agent halfway along the lane (relay candidate)

def one(job):
    label, kw, seed, dg, dp, bear, sp, T = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([]); eng.pop_events()
    need = (3 if RELAY else 2) + NBY
    ags = eng.dbg_keep_agents(need)
    if len(ags) < need: return dict(label=label, seed=seed, skip='agents')
    bait, guide = ags[0][0], ags[1][0]; relay = ags[2][0] if RELAY else None
    bys = [a[0] for a in ags[(3 if RELAY else 2):]]
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
        if (NOLINE or line_ok(gsx, gsy, ox, oy)) and line_ok(px, py, gsx, gsy): found = (gsx, gsy, px, py, phi); break
    if not found: return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='no_line')
    gsx, gsy, px, py, phi = found
    eng.dbg_set_agent(bait, gx, gy, math.atan2(-ay, -ax) + math.pi, 400., 10., 20., 500., 100., 400., 1.57, 1000.)
    eng.dbg_set_agent(guide, gsx, gsy, math.atan2(py - gsy, px - gsx), GE, float(sp), float(min(40, 2 * sp)), 800., 100., 400., 1.57, 1000.)
    import random as _r
    rng = _r.Random(seed * 1000 + int(dg) + int(dp) * 7 + int(bear) * 13 + int(sp))
    for b_ in bys:   # bystanders: ordinary colony members near the guide
        for _ in range(40):
            a_ = rng.uniform(0, 2 * math.pi); d_ = rng.uniform(40, 160); bx, by = gsx + d_ * math.cos(a_), gsy + d_ * math.sin(a_)
            if eng.dbg_free(bx - 5, by - 5, 10): break
        else: return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='by_pos')
        eng.dbg_set_agent(b_, bx, by, rng.uniform(0, 2 * math.pi), 250., 10., 20., 600., 100., 400., 1.57, 1000.)
    if RELAY:
        rx, ry = mx + dx * dg * 0.45, my + dy * dg * 0.45
        if not eng.dbg_free(rx - 5, ry - 5, 10): return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='relay_pos')
        eng.dbg_set_agent(relay, rx, ry, math.atan2(gsy - ry, gsx - rx), 300., float(sp), float(min(40, 2 * sp)), 800., 100., 400., 1.57, 1000.)
    if eng.dbg_pred_blocked(px, py): return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='pred_pos')
    # settle after teleporting: observations refresh and the policy's poses are re-synced twice (odometry/VO would drift)
    eng.dbg_freeze(([bait, guide, relay] if RELAY else [bait, guide]) + bys)
    for _ in range(3):
        eng.dbg_true_poses(); eng.run_policy(1e9, eng.info()['time'] + 0.1)
    eng.dbg_true_poses(); eng.dbg_freeze([bait, relay] if RELAY else [bait])
    if RELAY:   # keep the relay frozen (ineligible) until the policy has chosen the intended guide, then release it
        for _ in range(2):
            eng.run_policy(1e9, eng.info()['time'] + 0.1); eng.dbg_true_poses()
        eng.dbg_freeze([bait])
    # predators are placed only after the warm-up (a predator placed before it closes 45+ units on the frozen guide)
    if not eng.dbg_add_predator(px, py, math.atan2(gsy - py, gsx - px), 200., False): return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='pred_pos')
    extra_ok = 0; x_ang = []
    for k in range(NPRED - 1):   # edge case 2: more predators around the guide
        for _ in range(40):
            a_ = rng.uniform(0, 2 * math.pi); d_ = rng.uniform(60, 140); qx, qy = gsx + d_ * math.cos(a_), gsy + d_ * math.sin(a_)
            if not eng.dbg_pred_blocked(qx, qy) and eng.dbg_add_predator(qx, qy, math.atan2(gsy - qy, gsx - qx), 200., False):
                extra_ok += 1
                # angle between guide->trap-lane direction and guide->extra predator (0 = ahead on the way to the trap, 180 = behind)
                tl = math.atan2(oy - gsy, ox - gsx); x_ang.append(round(abs(math.degrees((a_ - tl + math.pi) % (2 * math.pi) - math.pi)))); x_ang.append(round(d_))
                break
    if extra_ok < NPRED - 1: return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, skip='pred2_pos')
    # step 2: K predators already held at the mouth (awake, full energy), spread across the lane just outside the mouth
    held_ok = 0
    for k in range(HELD):
        off = (k % 5 - 2) * 4.0; back = 12. + 8. * (k // 5)
        hx, hy = mx + ax * back - ay * off, my + ay * back + ax * off
        if eng.dbg_add_predator(hx, hy, math.atan2(gy - hy, gx - hx), 200., False): held_ok += 1
    t0 = eng.info()['time']; e0 = GE; killed = {}; dmin = 1e9; near = 0.; t_del = None; states = []; g_last = None; g_emin = GE; slow_ticks = 0; all_ticks = 0
    prev_g = None
    while eng.info()['time'] < t0 + T:
        eng.run_policy(t0 + T, eng.info()['time'] + (0.1 if TRACE else 0.5))
        if TRACE:
            ag_ = {a[0]: a for a in eng.agents()}; pr_ = eng.predators(); rr = eng.dbg_roles(); pi_ = eng.dbg_pred_info()
            g_ = ag_.get(guide)
            if g_ and pr_:
                q = min(pr_, key=lambda q: math.hypot(q[0] - g_[1], q[1] - g_[2]))
                mv = math.hypot(g_[1] - prev_g[0], g_[2] - prev_g[1]) if prev_g else 0.
                wall = not eng.dbg_free(g_[1] - 12, g_[2] - 12, 24)
                print(f"t{eng.info()['time']-t0:5.1f} st {rr[0][5] if rr else -1} g ({g_[1]:.0f},{g_[2]:.0f}) hd {g_[3]:+.2f} e {g_[5]:.0f} moved {mv:4.1f} wall12 {int(wall)} | p ({q[0]:.0f},{q[1]:.0f}) d {math.hypot(q[0]-g_[1], q[1]-g_[2]):5.1f} mode {pi_[0][3] if pi_ else -1} | g->mouth {math.hypot(g_[1]-mx, g_[2]-my):4.0f} p->mouth {math.hypot(q[0]-mx, q[1]-my):4.0f}", flush=True)
                prev_g = (g_[1], g_[2])
            elif not g_:
                print(f"t{eng.info()['time']-t0:5.1f} guide dead; pred->mouth {min((math.hypot(q[0]-mx, q[1]-my) for q in pr_), default=-1):.0f}", flush=True)
        for kind, t, a, age, en in eng.pop_events():
            if kind == 'predator': killed[a] = round(t - t0, 1)
        agn = {a[0]: a for a in eng.agents()}
        if guide in agn:
            ga = agn[guide]; g_emin = min(g_emin, ga[5])
            pr_ = eng.predators(); dpg = min((math.hypot(q[0] - ga[1], q[1] - ga[2]) for q in pr_), default=1e9)
            qn = min(pr_, key=lambda q: math.hypot(q[0] - ga[1], q[1] - ga[2])) if pr_ else None
            gb = eng.dbg_biome(ga[1], ga[2]); pb = eng.dbg_biome(qn[0], qn[1]) if qn else -1
            slow_ticks += 1 if gb in (1, 2, 4) else 0; all_ticks += 1
            g_last = (round(ga[5], 1), round(math.hypot(ga[1] - mx, ga[2] - my)), round(dpg, 1), gb, pb)
        pr = eng.predators()
        if pr:
            dmin = min(dmin, math.hypot(pr[0][0] - gx, pr[0][1] - gy))
            dm = math.hypot(pr[0][0] - mx, pr[0][1] - my)   # pr[0] is the delivered predator (added first)
            near = near + 0.5 if dm < 25 else 0.
            if near >= 3. and t_del is None and bait not in killed: t_del = round(eng.info()['time'] - t0, 1)
        r = eng.dbg_roles(); states.append(r[0][5] if r else -1)
        if bait in killed: break
        if t_del is not None and eng.info()['time'] - t0 > t_del + 10: break
    ag = {a[0]: a for a in eng.agents()}
    prs = eng.predators(); held_end = sum(1 for p_ in prs if math.hypot(p_[0] - mx, p_[1] - my) < 30)
    return dict(label=label, seed=seed, dg=dg, dp=dp, bear=bear, speed=sp, delivered=int(t_del is not None and bait not in killed), t_deliver=t_del,
                held0=HELD, held_placed=held_ok, held_end=held_end, npred=len(prs),
                guide_alive=int(guide in ag), t_guide_died=killed.get(guide), bait_alive=int(bait in ag), used=round(e0 - ag[guide][5], 1) if guide in ag else None,
                dmin_bait=round(dmin, 1), states=''.join(str(x) for x in states[::4]), site=[round(gx), round(gy), round(ov), rear], phi=phi,
                ge=GE, g_emin=round(g_emin, 1), g_last=g_last, x_ang=x_ang, g_slow=round(slow_ticks / max(1, all_ticks), 2), npred0=NPRED, nby=NBY, by_dead=sum(1 for b_ in bys if b_ in killed),
                held_all=sum(1 for p_ in prs if math.hypot(p_[0] - mx, p_[1] - my) < 30))

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
