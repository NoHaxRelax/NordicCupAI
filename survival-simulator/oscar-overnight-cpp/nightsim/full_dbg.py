"""Trace the trap machinery in a full game with predators. usage: full_dbg.py SEED CFG LABEL [T]"""
import sys, json, math, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import nightsim
seed, cfg, lab = int(sys.argv[1]), sys.argv[2], sys.argv[3]; T = float(sys.argv[4]) if len(sys.argv) > 4 else 900.
kw = json.load(open(cfg))[lab]
sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine; sim.step([]); eng.pop_events()
eng.policy_init(nightsim.seed_key(seed), kw)
t = 0.; STEP = float(sys.argv[5]) if len(sys.argv) > 5 else 10.
while eng.agents() and eng.info()['time'] < T:
    eng.run_policy(T, eng.info()['time'] + STEP)
    ev = [e for e in eng.pop_events() if e[0] == 'predator']
    ags = eng.agents(); prs = eng.predators(); tr = eng.dbg_trap(); roles = eng.dbg_roles(); sites = eng.dbg_sites(); ps = eng.dbg_pseen()
    held = sum(1 for p in prs if tr and math.hypot(p[0]-tr[0], p[1]-tr[1]) < 35)
    nseen = sum(len(x[1]) for x in ps if isinstance(x[1], list))
    main = max(roles, key=lambda r: r[1]) if roles else None
    if STEP < 10 and not (main and (main[4] >= 0 or main[2] >= 0 or main[3] >= 0)): continue
    gi = main[4] if main else -1; ga = [a for a in ags if a[0] == gi]
    bi = main[2] if main else -1; ba = [a for a in ags if a[0] == bi]; ri = main[3] if main else -1; ra = [a for a in ags if a[0] == ri]
    bpos = f"bait {bi} ({ba[0][1]:.0f},{ba[0][2]:.0f}) e{ba[0][5]:.0f} age{ba[0][4]:.0f} dgoal {math.hypot(ba[0][1]-tr[2], ba[0][2]-tr[3]):.0f}" if ba and tr else f"bait {bi}"
    rpos = f"rep {ri} ({ra[0][1]:.0f},{ra[0][2]:.0f}) e{ra[0][5]:.0f} dgoal {math.hypot(ra[0][1]-tr[2], ra[0][2]-tr[3]):.0f}" if ra and tr else f"rep {ri}"
    if os.environ.get('BAITTRACE'):
        kp = eng.dbg_keeper(); kid = kp[0] if kp else -1; ka = [a for a in ags if a[0] == kid]; kpp = eng.dbg_pose(kid) if kid >= 0 else None
        ktxt = f"keeper {kid} true ({ka[0][1]:.0f},{ka[0][2]:.0f}) pol ({kpp[0]:.0f},{kpp[1]:.0f}) e{ka[0][5]:.0f} d_rear {math.hypot(ka[0][1]-kp[2], ka[0][2]-kp[3]):.0f} spawn {kp[1]}" if ka and kpp else f"keeper {kid}"
        cid = bi if bi >= 0 else ri; cpp = eng.dbg_pose(cid) if cid >= 0 else None
        ctxt = f"bait/rep pol ({cpp[0]:.0f},{cpp[1]:.0f}) grp {cpp[3]} anch {cpp[4]}" if cpp else ""
        print(f"t{eng.info()['time']:6.1f} alive {len(ags):2d} | {ktxt} | {bpos} | {rpos} | {ctxt} | held {held}" if tr else f"t{eng.info()['time']:6.1f} no trap"); continue
        print(f"t{eng.info()['time']:6.1f} alive {len(ags):2d} preds {len(prs):2d} | {bpos} | {rpos} | policy pose of bait/rep ({main[13]:.0f},{main[14]:.0f}) | held {held} | goal ({tr[2]:.0f},{tr[3]:.0f}) mouth ({tr[0]:.0f},{tr[1]:.0f})" if tr else f"t{eng.info()['time']:6.1f} no trap"); continue
    gpos = f"({ga[0][1]:.0f},{ga[0][2]:.0f}) e{ga[0][5]:.0f}" if ga else 'dead'
    dpred = min((math.hypot(p[0]-ga[0][1], p[1]-ga[0][2]) for p in prs), default=-1) if ga else -1
    pp = min(prs, key=lambda p: math.hypot(p[0]-ga[0][1], p[1]-ga[0][2])) if ga and prs else None
    pinfo = eng.dbg_pred_info(); pi_near = pinfo[prs.index(pp)] if pp else None
    print(f"t{eng.info()['time']:6.1f} alive {len(ags):2d} preds {len(prs):2d} eaten {len(ev)} | bait {main[2] if main else None} guide {gi} st {main[5] if main else None} dP(pol) {main[7] if main else 0:.0f} | guide true {gpos} pol ({main[9]:.0f},{main[10]:.0f}) | nearest pred true ({pp[0]:.0f},{pp[1]:.0f}) d {dpred:.0f}, pol target ({main[11]:.0f},{main[12]:.0f}) | pred targets d {pi_near[0]:.0f} mode {pi_near[3]} (guide at {dpred:.0f}) | held {held}" if ga and pp else f"t{eng.info()['time']:6.1f} guide {gi} {gpos}")
