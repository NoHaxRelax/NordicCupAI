"""Diagnostic only: absolute pose error of anchored groups and correction sizes."""
import os, sys, pathlib, math, collections, statistics
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
import orchard
from orchard import OrchardPolicy, wrap, norm, sub, add

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 200
corr = []
orig_observe = OrchardPolicy._observe
def observe(self, m, s):
    before = m.pose.p
    orig_observe(self, m, s)
    if before != m.pose.p and (m.last_action is None or m.last_action[0] <= 0):
        corr.append(('nomove', norm(sub(m.pose.p, before))))
    elif before != m.pose.p:
        corr.append(('move', norm(sub(m.pose.p, before))))
OrchardPolicy._observe = observe

import json; KW = json.loads(os.environ.get('ORCHARD_KW', '{}'))
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed, **KW)
state = sim.step([])
errs = []; tree_errs = []; n_anch = 0; n_tot = 0
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time'])
    # after the policy call, poses are current for this tick's observations
    truth = {a.agent_id: (a.x, a.y, a.direction) for a in env.agents}
    for aid, m in pol.minds.items():
        g = pol.groups[m.group]; n_tot += 1
        if not g.anchored or aid not in truth: continue
        n_anch += 1
        tx, ty, td = truth[aid]
        errs.append(math.hypot(m.pose.p[0]-tx, m.pose.p[1]-ty))
        if abs(wrap(m.pose.theta-td)) > 1e-6: errs.append(1e6)
    if int(state['sim_time']*10) % 100 == 0:
        for g in pol.groups.values():
            if not g.anchored: continue
            for t in g.trees.values():
                if t.dead: continue
                d = min((math.hypot(t.p[0]-tr.x, t.p[1]-tr.y) for tr in env.trees), default=None)
                if d is not None: tree_errs.append(d)
    state = sim.step(actions)
errs_f = [e for e in errs if e < 1e5]
q = lambda xs: [round(v, 2) for v in statistics.quantiles(xs, n=10)] if len(xs) > 10 else xs
print('anchored agent-ticks', n_anch, 'of', n_tot, 'heading mismatches', sum(1 for e in errs if e >= 1e5))
print('pos err quantiles (10%..90%)', q(errs_f), 'max', round(max(errs_f), 1) if errs_f else None, '>5:', sum(1 for e in errs_f if e > 5))
print('tree map err quantiles', q(tree_errs), '>10:', sum(1 for e in tree_errs if e > 10), 'of', len(tree_errs))
by = collections.defaultdict(list)
for k, v in corr: by[k].append(v)
for k, v in by.items(): print('corrections', k, len(v), 'size quantiles', q(v))
print('metrics', {k: v for k, v in pol.metrics.items() if k in ('pose_corrections', 'deflections', 'stuck_events', 'anchors', 'merges')})

print('SUMMARY', json.dumps(dict(kw=KW, seed=seed, bad5=round(sum(1 for e in errs_f if e > 5)/max(1,len(errs_f)),4), tree_bad=round(sum(1 for e in tree_errs if e > 10)/max(1,len(tree_errs)),4), stuck=pol.metrics['stuck_events'])))
