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
    """engine-truth sink detector: npred awake predators at uniform random free spots, no agents, T seconds; a sink is a
    cluster (30) of predators that stayed within R of their end point for the last `settle` s. -> [[x, y, n_captured]]"""
    settle = settle or min(300., T / 2)
    eng.dbg_keep_agents(0)
    rng = random.Random(seed * 7919 + 3); placed = 0
    for _ in range(npred * 50):
        if placed >= npred: break
        x, y = rng.uniform(20, 1580), rng.uniform(20, 1180)
        if eng.dbg_add_predator(x, y, rng.uniform(0, 2 * math.pi), rng.uniform(100, 200), False): placed += 1
    n0 = len(eng.predators()); t = eng.info()['time']; tend = t + T; hist = []
    while t < tend - 1e-6:
        eng.dbg_step(50); t = eng.info()['time']
        hist.append([(p[0], p[1]) for p in eng.predators()[:n0]])
    k = int(settle / 5.); end = hist[-1]; pts = []
    for i in range(n0):
        ex, ey = end[i]
        if all(math.hypot(h[i][0] - ex, h[i][1] - ey) <= R for h in hist[-k:]):
            if all(math.hypot(h[i][0] - ex, h[i][1] - ey) < 1. for h in hist): continue   # never moved: embedded start, not a sink
            pts.append((ex, ey))
    sinks = []
    for x, y in pts:
        for s in sinks:
            if math.hypot(s[0] - x, s[1] - y) < 30.: s[3].append((x, y)); break
        else: sinks.append([x, y, 0, [(x, y)]])
    out = []
    for s in sinks:
        xs = [p[0] for p in s[3]]; ys = [p[1] for p in s[3]]
        out.append([round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1), len(s[3])])
    out.sort(key=lambda s: -s[2])
    return out, dict(n=n0, stuck=len(pts))


def drop_test(seed, sinks, n, T):
    """hold test: n predators dropped at each sink point (jitter 6, random heading, energy 60-200), no agents, T s;
    -> per sink [held_all_T (never left R of the sink point), at_sink_end]"""
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    sim.step([]); eng.dbg_keep_agents(0)
    rng = random.Random(seed * 31 + 7); owner = []
    for si, s in enumerate(sinks):
        for k in range(n):
            for _ in range(40):
                x, y = s[0] + rng.uniform(-6, 6), s[1] + rng.uniform(-6, 6)
                if eng.dbg_add_predator(x, y, rng.uniform(0, 2 * math.pi), rng.uniform(60, 200), False): owner.append(si); break
    held = [True] * len(owner); t = eng.info()['time']; tend = t + T
    while t < tend - 1e-6:
        eng.dbg_step(10); t = eng.info()['time']; P = eng.predators()
        for i, si in enumerate(owner):
            if held[i] and math.hypot(P[i][0] - sinks[si][0], P[i][1] - sinks[si][1]) > R: held[i] = False
    P = eng.predators(); out = []
    for si in range(len(sinks)):
        idx = [i for i, o in enumerate(owner) if o == si]
        out.append([sum(held[i] for i in idx), sum(1 for i in idx if math.hypot(P[i][0] - sinks[si][0], P[i][1] - sinks[si][1]) <= R), len(idx)])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='wander'); ap.add_argument('--configs', default='{"none":{}}')
    ap.add_argument('--seeds', nargs='+', required=True); ap.add_argument('--npred', type=int, default=40)
    ap.add_argument('--T', type=float, default=900.); ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    jobs = [(l, kw, s, a.mode, a.npred, a.T) for l, kw in cfgs.items() for s in parse_seeds(a.seeds)]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} in {time.time()-t0:.0f}s', flush=True)
