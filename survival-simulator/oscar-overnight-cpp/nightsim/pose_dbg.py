"""Pose-error probe: mean/median |policy pose - true pose| over living agents every 25 s. usage: pose_dbg.py SEED CFG LABEL T [predators]"""
import sys, json, math, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import nightsim
seed, cfg, lab, T = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4]); preds = len(sys.argv) > 5
kw = json.load(open(cfg))[lab]
sim = nightsim.SimulationCore(seed=seed, predators=preds); eng = sim._engine; sim.step([]); eng.pop_events()
eng.policy_init(nightsim.seed_key(seed), kw)
while eng.agents() and eng.info()['time'] < T:
    eng.run_policy(T, eng.info()['time'] + 25.)
    errs = []; dx = []; dy = []
    for a in eng.agents():
        p = eng.dbg_pose(a[0])
        if not p: continue
        errs.append(math.hypot(p[0] - a[1], p[1] - a[2])); dx.append(a[1] - p[0]); dy.append(a[2] - p[1])
    if errs: print(f"t{eng.info()['time']:5.0f} n{len(errs):3d} err mean {st.mean(errs):6.1f} median {st.median(errs):6.1f} | median true-pol dx {st.median(dx):6.1f} dy {st.median(dy):6.1f} | n with err<10: {sum(e < 10 for e in errs)}")
