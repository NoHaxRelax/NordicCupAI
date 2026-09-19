"""Stuck-spot study: where do predators get permanently stuck under the unmodified predator AI? (engine truth; runs on pods)
usage: stuckspot.py --mode wander|game --seeds 1-64 --out rows.jsonl [--configs CFG] [--npred 40] [--T 900]
wander: fresh map, all agents removed, --npred awake predators at uniform random free spots (plus the engine's own spawns),
        stepped --T seconds with no agents (pure edge-avoid / random-walk behaviour).
game:   full game with the policy config (--configs, one label) and predators on; every predator tracked.
Tracking: positions sampled every 1 s; a stuck episode = the predator stays within R (30) of its position at the episode
start for >= MIN (30) s. Row: label, seed, mode, obstacles (wander only), per-predator seconds tracked, events
[pid, t0, dur, cx, cy, spread, censored, modes_before(1..4), modes_during(1..4), rest_frac, agent_dist_at_entry, exit_mode]
modes: 1 direct chase, 2 pivot chase, 3 edge-avoid, 4 random wander."""
import argparse, json, math, os, sys, time, pathlib, random
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds

R = float(os.environ.get('STUCK_R', '30')); MIN = float(os.environ.get('STUCK_MIN', '30'))


class Tracker:
    def __init__(self):
        self.st = {}      # pid -> dict(anchor, t0, pts, modes_before(deque of last 10), modes, rest, adist)
        self.events = []; self.tracked = {}

    def update(self, t, preds, info, agents):
        for i, (x, y, d, e, rest) in enumerate(preds):
            mode = info[i][3] if i < len(info) else 0
            if rest: mode = 0
            s = self.st.get(i)
            self.tracked[i] = self.tracked.get(i, 0) + 1
            if s is None:
                self.st[i] = s = dict(ax=x, ay=y, t0=t, pts=[(x, y)], hist=[], md=[0] * 5, adist=_adist(x, y, agents))
            elif math.hypot(x - s['ax'], y - s['ay']) > R:
                self._close(i, s, t, censored=0, exit_mode=mode)
                hist = (s['hist'] + [mode])[-10:]
                self.st[i] = s = dict(ax=x, ay=y, t0=t, pts=[(x, y)], hist=hist, md=[0] * 5, adist=_adist(x, y, agents))
                continue
            else:
                s['pts'].append((x, y))
            s['md'][mode] += 1
            if len(s['pts']) <= 1: pass
            s.setdefault('hist_before', list(s['hist']))
            s['hist'] = (s['hist'] + [mode])[-10:]

    def _close(self, i, s, t, censored, exit_mode):
        dur = t - s['t0']
        if dur >= MIN:
            xs = [p[0] for p in s['pts']]; ys = [p[1] for p in s['pts']]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            spread = max(math.hypot(px - cx, py - cy) for px, py in s['pts'])
            hb = s.get('hist_before', []); mb = [sum(1 for m in hb if m == k) for k in range(5)]
            n = max(1, sum(s['md']))
            self.events.append([i, round(s['t0']), round(dur), round(cx, 1), round(cy, 1), round(spread, 1), censored, mb,
                                s['md'], round(s['md'][0] / n, 2), s['adist'], exit_mode])

    def finish(self, t):
        for i, s in self.st.items(): self._close(i, s, t, censored=1, exit_mode=-1)


def _adist(x, y, agents):
    return round(min((math.hypot(a[1] - x, a[2] - y) for a in agents), default=1e4))


def one(job):
    label, kw, seed, mode, npred, T = job
    import nightsim
    t0w = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([]); eng.pop_events()
    obst = [[round(v) for v in o] for o in eng.obstacles()]
    tr = Tracker()
    if mode == 'wander':
        eng.dbg_keep_agents(0)
        rng = random.Random(seed * 7919 + 1)
        placed = 0
        for _ in range(npred * 50):
            if placed >= npred: break
            x, y = rng.uniform(20, 1580), rng.uniform(20, 1180)
            if eng.dbg_add_predator(x, y, rng.uniform(0, 2 * math.pi), rng.uniform(100, 200), False): placed += 1
        t = eng.info()['time']; tend = t + T
        while t < tend - 1e-6:
            eng.dbg_step(10); t = eng.info()['time']
            tr.update(t, eng.predators(), eng.dbg_pred_info(), [])
        tr.finish(t)
        return dict(label=label, seed=seed, mode=mode, T=T, npred=len(eng.predators()), obst=obst,
                    tracked=[tr.tracked[i] for i in sorted(tr.tracked)], events=tr.events, wall=round(time.perf_counter() - t0w, 1))
    if mode == 'probe':
        sinks, info = probe(eng, seed, npred, T)
        drops = drop_test(seed, sinks, int(os.environ.get('STUCK_DROPS', '8')), float(os.environ.get('STUCK_DROP_T', '300')))
        return dict(label=label, seed=seed, mode=mode, obst=obst, sinks=[s + [d] for s, d in zip(sinks, drops)], probe=info,
                    wall=round(time.perf_counter() - t0w, 1))
    if mode == 'spec':
        return dict(label=label, seed=seed, mode=mode, specs=kill_specs(eng, seed, SINKS.get(seed, [])), wall=round(time.perf_counter() - t0w, 1))
    eng.policy_init(nightsim.seed_key(seed), kw)
    apos = []
    horizon = T
    while True:
        info = eng.info()
        if not eng.agents() or info['time'] >= horizon: break
        eng.run_policy(horizon, info['time'] + 1.0)
        eng.pop_events()
        ags = eng.agents(); tn = eng.info()['time']
        tr.update(tn, eng.predators(), eng.dbg_pred_info(), ags)
        if abs(tn % 25.) < 0.05 or abs(tn % 25. - 25.) < 0.05: apos.append([round(tn)] + [[round(a[1]), round(a[2])] for a in ags])
    t = eng.info()['time']; tr.finish(t)
    return dict(label=label, seed=seed, mode=mode, surv=round(t, 1), npred=len(eng.predators()), obst=obst,
                tracked=[tr.tracked[i] for i in sorted(tr.tracked)], events=tr.events, apos=apos, wall=round(time.perf_counter() - t0w, 1))


def probe(eng, seed, npred, T, settle=None):
    """engine-truth sink detector: npred awake predators at uniform random free spots, no agents, T seconds, then 40
    single ticks; a predator is captured when its awake positions repeat exactly with period <= 12 (a limit cycle of the
    deterministic edge-avoid + collision-deflection rule). Sinks = clusters (30) of captured predators' cycles.
    -> [[x, y, n_captured, period, heading_at_vertex, biome, walls_within_25, boundary]] (x, y = a real cycle vertex)"""
    eng.dbg_keep_agents(0)
    rng = random.Random(seed * 7919 + 3); placed = 0
    for _ in range(npred * 50):
        if placed >= npred: break
        x, y = rng.uniform(20, 1580), rng.uniform(20, 1180)
        if eng.dbg_add_predator(x, y, rng.uniform(0, 2 * math.pi), rng.uniform(100, 200), False): placed += 1
    n0 = len(eng.predators())
    eng.dbg_step(int(T * 10) - 40); tr = []
    for _ in range(40):
        eng.dbg_step(1); tr.append(eng.predators()[:n0])
    ob = eng.obstacles(); cyc = []
    for i in range(n0):
        seq = [t[i] for t in tr if not t[i][4]]
        if len(seq) < 20: continue
        for p in range(1, 13):
            if all(abs(seq[j][0] - seq[j - p][0]) < 1e-9 and abs(seq[j][1] - seq[j - p][1]) < 1e-9 for j in range(p, len(seq))):
                cyc.append((p, seq[-p:])); break
    sinks = []
    for p, vs in cyc:
        cx = sum(v[0] for v in vs) / p; cy = sum(v[1] for v in vs) / p
        for s in sinks:
            if math.hypot(s[0] - cx, s[1] - cy) < 30.: s[2].append((p, vs)); break
        else: sinks.append([cx, cy, [(p, vs)]])
    out = []
    for cx, cy, members in sinks:
        mx = sum(sum(v[0] for v in vs) / p for p, vs in members) / len(members); my = sum(sum(v[1] for v in vs) / p for p, vs in members) / len(members)
        best = min(((v, p) for p, vs in members for v in vs), key=lambda z: math.hypot(z[0][0] - mx, z[0][1] - my))
        (vx, vy, vd, ve, vr), p = best
        nw = sum(1 for o in ob if math.hypot(max(o[0] - vx, 0, vx - o[0] - o[2]), max(o[1] - vy, 0, vy - o[1] - o[3])) < 25)
        bnd = int(min(vx - 30, 1570 - vx, vy - 30, 1170 - vy) < 20)
        out.append([round(vx, 2), round(vy, 2), len(members), p, round(vd % (2 * math.pi), 3), eng.dbg_biome(vx, vy), nw, bnd])
    out.sort(key=lambda s: -s[2])
    return out, dict(n=n0, stuck=len(cyc))


def drop_test(seed, sinks, n, T):
    """hold test: at each sink vertex drop 1 predator in the exact cycle state (vertex + heading) and n-1 with jitter 6 and a
    random heading (energy 60-200), no agents, T s. -> per sink [exact_held, random_held_all_T, random_at_sink_end, n_random]"""
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([]); eng.dbg_keep_agents(0)
    rng = random.Random(seed * 31 + 7); owner = []
    for si, s in enumerate(sinks):
        owner.append((si, 1) if eng.dbg_add_predator(s[0], s[1], s[4], 150., False) else (si, -1))
        for k in range(n - 1):
            for _ in range(40):
                x, y = s[0] + rng.uniform(-6, 6), s[1] + rng.uniform(-6, 6)
                if eng.dbg_add_predator(x, y, rng.uniform(0, 2 * math.pi), rng.uniform(60, 200), False): owner.append((si, 0)); break
    owner = [o for o in owner if o[1] >= 0]
    held = [True] * len(owner); t = eng.info()['time']; tend = t + T
    while t < tend - 1e-6:
        eng.dbg_step(10); t = eng.info()['time']; P = eng.predators()
        for i, (si, ex) in enumerate(owner):
            if held[i] and math.hypot(P[i][0] - sinks[si][0], P[i][1] - sinks[si][1]) > R: held[i] = False
    P = eng.predators(); out = []
    for si in range(len(sinks)):
        ie = [i for i, o in enumerate(owner) if o == (si, 1)]; ir = [i for i, o in enumerate(owner) if o == (si, 0)]
        out.append([int(bool(ie) and held[ie[0]]), sum(held[i] for i in ir),
                    sum(1 for i in ir if math.hypot(P[i][0] - sinks[si][0], P[i][1] - sinks[si][1]) <= R), len(ir)])
    return out


SINKS = {}
SPEC_MINN = int(os.environ.get('STUCK_SPEC_MINN', '3')); KILLTEST = int(os.environ.get('STUCK_KILLTEST', '0')); LANE = float(os.environ.get('STUCK_LANE', '70'))


def kill_specs(eng, seed, sinks):
    """delivery spec per sink (captures >= SPEC_MINN): kill states (G, heading) near the cycle vertex from which the
    deterministic no-agent predator dynamics fall into a cycle within 30 of the vertex. Grid 3 units within 21 of the
    vertex x 24 headings, all dropped at once (predators do not interact), 350 ticks. A spec needs a straight lane: the
    points G - k u(heading), k = 5..LANE, predator-free. Robustness = success rate over the 3x3 grid neighbours x
    +-1 heading step. -> per sink [sink_index, vx, vy, n_candidates, n_success, best specs [Gx, Gy, th, robust, Lx, Ly]]"""
    eng.dbg_keep_agents(0)
    NH = 24; out = []
    todo = [(si, s) for si, s in enumerate(sinks) if s[2] >= SPEC_MINN]
    if not todo: return out
    base = len(eng.predators()); idx = {}
    for si, s in todo:
        vx, vy = s[0], s[1]
        for ix in range(-7, 8):
            for iy in range(-7, 8):
                x, y = vx + 3 * ix, vy + 3 * iy
                if 3 * math.hypot(ix, iy) > 21.01 or eng.dbg_pred_blocked(x, y): continue
                for h in range(NH):
                    if eng.dbg_add_predator(x, y, 2 * math.pi * h / NH, 200., False):
                        idx[(si, ix, iy, h)] = base; base += 1
    eng.dbg_step(150); hist = []
    for _ in range(20):
        eng.dbg_step(10); hist.append(eng.predators())
    ok = {}
    for k, i in idx.items():   # held: within 30 of the vertex for the last 200 ticks (a cycle; resting keeps it in place)
        si = k[0]; vx, vy = sinks[si][0], sinks[si][1]
        ok[k] = all(math.hypot(h[i][0] - vx, h[i][1] - vy) < 30 for h in hist)
    for si, s in todo:
        keys = [k for k in idx if k[0] == si]
        cands = []
        # the agent stands at A; the predator touches it (< 15) one move short, so the kill state is ~A - k u(th), k 8..15
        def okat(x, y, h):
            ix, iy = round((x - s[0]) / 3), round((y - s[1]) / 3)
            return ok.get((si, ix, iy, h % NH))
        for ix in range(-10, 11):
            for iy in range(-10, 11):
                ax, ay = s[0] + 3 * ix, s[1] + 3 * iy
                if 3 * math.hypot(ix, iy) > 30.01 or not eng.dbg_free(ax - 5, ay - 5, 10): continue
                for h in range(NH):
                    th = 2 * math.pi * h / NH; ux, uy = math.cos(th), math.sin(th)
                    vals = [okat(ax - k * ux, ay - k * uy, h + dh) for k in (8, 10, 12, 14) for dh in (-1, 0, 1)]
                    vals = [v for v in vals if v is not None]
                    if len(vals) < 6: continue
                    rob = sum(vals) / len(vals)
                    if rob < 0.3: continue
                    if any(eng.dbg_pred_blocked(ax - d * ux, ay - d * uy) for d in range(15, int(LANE) + 1, 5)): continue
                    cands.append([round(ax, 1), round(ay, 1), round(th, 4), round(rob, 3), round(ax - LANE * ux, 1), round(ay - LANE * uy, 1)])
        cands.sort(key=lambda c: -c[3])
        if KILLTEST > 0:
            scored = []
            for c in cands[:KILLTEST]:
                scored.append(c + [kill_test(seed, s, c)])
            scored.sort(key=lambda c: (-c[6], -c[3]))
            cands = scored
        out.append([si, s[0], s[1], len(keys), sum(1 for k in keys if ok[k]), cands[:5]])
    return out


def kill_test(seed, sink, c):
    """agent frozen at G, predator approaching along the lane (4 variants: distance 50/70 on the axis, 60 at +-8 off
    the axis), heading at G; after the kill 300 ticks without agents. -> fraction ending in a cycle within 30 of the vertex"""
    import nightsim
    gx, gy, th = c[0], c[1], c[2]; ux, uy = math.cos(th), math.sin(th); good = 0; n = 0
    for D, off in ((50, 0), (70, 0), (60, 8), (60, -8)):
        sx, sy = gx - D * ux - off * uy, gy - D * uy + off * ux
        sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
        sim.step([]); ags = eng.dbg_keep_agents(1)
        if eng.dbg_pred_blocked(sx, sy) or not ags: continue
        eng.dbg_set_agent(ags[0][0], gx, gy, th + math.pi, 300., 10., 20., 800., 100., 400., 1.57, 1000.)
        if not eng.dbg_add_predator(sx, sy, math.atan2(gy - sy, gx - sx), 200., False): continue
        n += 1
        for _ in range(60):
            eng.dbg_step(1)
            if not eng.agents(): break
        if eng.agents(): continue
        eng.dbg_step(100); hold = True
        for _ in range(20):
            eng.dbg_step(10); p = eng.predators()[0]
            if math.hypot(p[0] - sink[0], p[1] - sink[1]) >= 30: hold = False; break
        good += hold
    return round(good / max(1, n), 3)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='wander'); ap.add_argument('--configs', default='{"none":{}}')
    ap.add_argument('--seeds', nargs='+', required=True); ap.add_argument('--npred', type=int, default=40)
    ap.add_argument('--T', type=float, default=900.); ap.add_argument('--sinks', default=''); ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    if a.sinks:
        for l in open(a.sinks):
            r = json.loads(l); SINKS[r['seed']] = r['sinks']
    jobs = [(l, kw, s, a.mode, a.npred, a.T) for l, kw in cfgs.items() for s in parse_seeds(a.seeds)]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
