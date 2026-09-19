"""Debug one guide scenario: prints site geometry vs true obstacles, then a per-0.5 s trace. usage: guide_dbg.py SEED DG DP BEAR SPEED"""
import sys, json, math, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import nightsim
from nightsim.trapsite import grade
seed, dg, dp, bear, sp = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
PHI = [int(sys.argv[6])] if len(sys.argv) > 6 else [0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60]
CFG = sys.argv[7] if len(sys.argv) > 7 else '/workspace/night/cfg-r21.json'; LAB = sys.argv[8] if len(sys.argv) > 8 else 'r21s0c2'
kw = json.load(open(CFG))[LAB]
sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
sim.step([]); eng.pop_events()
ags = eng.dbg_keep_agents(2); bait, guide = ags[0][0], ags[1][0]
cfg = dict(kw, trap_mode=3, merge_anchored=1, pred_mode=1, pred_share=1, no_spawn=1, trap_bait_fixed=bait, trap_start=0)
eng.policy_init(nightsim.seed_key(seed), cfg)
eng.run_policy(1e9, eng.info()['time'] + 0.1); print('true poses', eng.dbg_true_poses(), 'walls loaded groups', eng.dbg_load_walls())
eng.run_policy(1e9, eng.info()['time'] + 0.1)
sites = eng.dbg_sites(); print('sites', len(sites))
for st in sites[:3]:
    gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = st
    free, blocked, lane = grade(eng, gx, gy, ox, oy)
    print(f'  goal ({gx:.1f},{gy:.1f}) mouth ({mx:.1f},{my:.1f}) out ({ox:.1f},{oy:.1f}) overlap {ov:.1f} gap {gap:.1f} rear {rear} -> free {free} blocked {blocked} lane {lane}')
    obs = [o for o in eng.obstacles() if o[0]-40 < gx < o[0]+o[2]+40 and o[1]-40 < gy < o[1]+o[3]+40]
    print('   obstacles near goal:', [tuple(round(v, 1) for v in o) for o in obs])
    bad = [(r, round(math.degrees(a))) for r in (0., 5., 10., 14.9) for a in [k*math.pi/18 for k in range(36)] if not eng.dbg_pred_blocked(gx + r*math.cos(a), gy + r*math.sin(a))]
    print('   unblocked points:', bad[:12])
gid, gx, gy, mx, my, ox, oy, ov, gap, rear, nw = sites[0]
ax, ay = (mx - gx), (my - gy); n = math.hypot(ax, ay); ax, ay = ax / n, ay / n
def clear(x0, y0, x1, y1): return all(not eng.dbg_pred_blocked(x0 + (x1-x0)*i/40, y0 + (y1-y0)*i/40) for i in range(41))
found = None
for phi in PHI:
    f = math.radians(phi); dx, dy = ax * math.cos(f) - ay * math.sin(f), ax * math.sin(f) + ay * math.cos(f)
    gsx, gsy = mx + dx * dg, my + dy * dg
    b = math.radians(bear); px, py = gsx + dp * (dx * math.cos(b) - dy * math.sin(b)), gsy + dp * (dx * math.sin(b) + dy * math.cos(b))
    if not eng.dbg_free(gsx - 5, gsy - 5, 10) or eng.dbg_pred_blocked(px, py): continue
    if clear(gsx, gsy, ox, oy) and clear(px, py, gsx, gsy): found = (gsx, gsy, px, py, phi); break
print('found', found)
gsx, gsy, px, py, phi = found
eng.dbg_set_agent(bait, gx, gy, 0., 400., 10., 20., 500., 100., 400., 1.57, 1000.)
eng.dbg_set_agent(guide, gsx, gsy, math.atan2(py - gsy, px - gsx), 300., sp, min(40, 2*sp), 800., 100., 400., 1.57, 1000.)
print('pred added', eng.dbg_add_predator(px, py, math.atan2(gsy - py, gsx - px), 200., False))
eng.dbg_freeze([bait, guide])
for _ in range(3):
    eng.dbg_true_poses(); eng.run_policy(1e9, eng.info()['time'] + 0.1)
eng.dbg_true_poses(); eng.dbg_freeze([bait])
t0 = eng.info()['time']
for k in range(60):
    eng.run_policy(t0 + 60, eng.info()['time'] + 0.2)
    ev = [e for e in eng.pop_events() if e[0] != 'fruit']
    ag = {a[0]: a for a in eng.agents()}; pr = eng.predators()
    g = ag.get(guide); p = pr[0] if pr else None
    r = eng.dbg_roles(); ps = eng.dbg_pseen(); pi = eng.dbg_pred_info()
    d = math.hypot(g[1]-p[0], g[2]-p[1]) if g and p else None
    dm = math.hypot(p[0]-mx, p[1]-my) if p else None
    b_ = ag.get(bait); dbait = math.hypot(p[0]-b_[1], p[1]-b_[2]) if b_ and p else None
    if g and p:
        print(f"t{eng.info()['time']-t0:5.1f} guide ({g[1]:.0f},{g[2]:.0f}) e {g[5]:.0f} | pred ({p[0]:.0f},{p[1]:.0f}) e {p[3]:.0f} rest {p[4]} | d {d:.0f} d_mouth {dm:.0f} d_bait {dbait:.0f} state {r[0][5] if r else None} | pred sees {pi[0][0]:.0f} look {pi[0][1]:.2f} mode {pi[0][3]} {ev}")
    else:
        print(f"t{eng.info()['time']-t0:5.1f} guide dead | pred ({p[0]:.0f},{p[1]:.0f}) e {p[3]:.0f} rest {p[4]} d_mouth {dm:.0f} d_bait {dbait:.0f} bait alive {b_ is not None} | pred sees {pi[0][0]:.0f} mode {pi[0][3]} {ev}" if p else 'no predator')
    if not b_: break
