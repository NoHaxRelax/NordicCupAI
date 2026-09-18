"""Find the first internal divergence between the Python and native orchard policy.
    python fastsim/debug_policy.py SEED [--predators]"""
import os, sys, json, pathlib, math
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor' / 'survival-simulator'), str(HERE / 'policy')]
import fastsim, orchard_ref as orchard
seed = int(sys.argv[1]); preds = '--predators' in sys.argv
cfg = json.load(open(HERE / 'policy' / 'best-config.json'))
sim = fastsim.SimulationCore(seed=seed, predators=preds); e = sim._engine
e.policy_init(fastsim.seed_key(seed), cfg); pol = orchard.OrchardPolicy(seed=seed, **cfg)
def pym():
    out = []
    for aid, m in pol.minds.items():
        out.append((aid, m.group, m.pose.p[0], m.pose.p[1], m.pose.theta, m.post, m.fruit, m.old,
                    None if m.explore is None else tuple(m.explore[0]), None if m.watch is None else tuple(m.watch[0]),
                    len(m.edges), m.best_d, m.energy_prev, len(m.prev_marks), len(m.hear_hist),
                    m.prev_pose.theta if m.prev_pose is not None else 0.0))
    return out
def pyg():
    return [(gid, g.anchored, [(tid, t.p[0], t.p[1], t.first, t.last, t.dead, []) for tid, t in g.trees.items()],
             [(fid, f.p[0], f.p[1], f.born_lo, f.born_hi) for fid, f in g.fruits.items()], len(g.cells), g.next_tree, g.next_fruit)
            for gid, g in pol.groups.items()]
state = sim.step([]); n = 0
while state['num_agents']:
    py = pol(state['observations'], state['sim_time']); nat = e.policy_act()
    a, b = pym(), e.policy_minds()
    ga_, gb_ = pyg(), e.policy_groups()
    if ga_ != gb_ and a == b:
        for x, y in zip(ga_, gb_):
            if x != y:
                print('step', n, 't', state['sim_time'], 'group', x[0], 'differs first (minds equal); next', x[5:], y[5:], 'ntrees', len(x[2]), len(y[2]))
                for u, v in zip(x[2], y[2]):
                    if u != v: print('  tree', u, '\n       ', v); break
                for u, v in zip(x[3], y[3]):
                    if u != v: print('  fruit', u, '\n        ', v); break
                break
        break
    if a != b:
        for x, y in zip(a, b):
            if x != y:
                print('step', n, 't', state['sim_time'], 'first mind diff\n py ', x, '\n nat', y); break
        else: print('step', n, 'minds len', len(a), len(b))
        ga, gb = pyg(), e.policy_groups()
        for x, y in zip(ga, gb):
            if x != y:
                print(' group', x[0], 'anchored', x[1], y[1], 'ntrees', len(x[2]), len(y[2]), 'nfruits', len(x[3]), len(y[3]), 'cells', x[4], y[4], 'next', x[5:], y[5:])
                for u, v in zip(x[2], y[2]):
                    if u != v: print('  tree', u, '\n       ', v); break
                for u, v in zip(x[3], y[3]):
                    if u != v: print('  fruit', u, '\n        ', v); break
                break
        break
    state = sim.step(py); n += 1
else:
    print('no divergence; steps', n)
