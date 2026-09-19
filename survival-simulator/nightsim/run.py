"""Batch runner for the private native engine+policy (nightsim). One compact JSON line per game, no replays.
Usage: python nightsim/run.py --configs CFG.json|'{"lab":{...}}' --seeds 1-64 [--predators] --workers 32 --out rows.jsonl
Row: label, seed, surv, score, fruit, eaten, peak, created, pdeaths (predator kills), sdeaths, penalty,
trees/fruits/alive at death (last 50 s sample), traj (alive,trees every 250 s)."""
import argparse, json, os, sys, time, pathlib, hashlib, platform
from multiprocessing import Pool
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

def parse_seeds(items):
    out = []
    for it in items:
        for part in str(it).split(','):
            if '-' in part:
                a, b = part.split('-'); out += list(range(int(a), int(b)+1))
            elif part: out.append(int(part))
    return out

DIAG_FROM = float(os.environ.get('NIGHT_DIAG_FROM', '0'))
TAIL = int(os.environ.get('NIGHT_TAIL', '0'))   # >0: keep the last TAIL 10-s snapshots (alive, old, meanE, minE, ages) before the end  # >0: per-fruit fate diagnostics after this time (engine truth, analysis only)

def fruit_diag(eng, horizon, stop_at, F, out):
    # step in 1 s chunks; F: fid -> [spawn_t, last_age, min_agent_dist, x, y]
    import math
    t = eng.info()['time']; peak = 0
    while eng.agents() and t < min(horizon, stop_at) - 1e-6:
        _, cp = eng.run_policy(horizon, min(stop_at, t + 1.0)); peak = max(peak, cp)
        t = eng.info()['time']; ags = eng.agents(); cur = {}
        for (fid, x, y, en, age, rad, _) in eng.fruits():
            d = min((math.hypot(a[1]-x, a[2]-y) for a in ags), default=1e9)
            if fid in F: F[fid][1] = age; F[fid][2] = min(F[fid][2], d)
            else: F[fid] = [t-age, age, d, x, y]
            cur[fid] = 1
        for fid in [k for k in F if k not in cur]:
            spawn_t, age, dmin, x, y = F.pop(fid)
            out.append((round(spawn_t), 'rot' if age >= 49.4 else 'eat', round(dmin), round(age, 1)))
    return peak

def one(job):
    label, kw, seed, horizon, predators, sample = job
    import nightsim
    t0 = time.perf_counter()
    sim = nightsim.SimulationCore(seed=seed, predators=predators)
    eng = sim._engine
    state = sim.step([])
    fe = 0.; eaten = 0; pd = 0; sd = 0; pen = 0.; kills = []; deaths = []
    def events():
        nonlocal fe, eaten, pd, sd, pen
        for kind, t, aid, age, energy in eng.pop_events():
            if kind == 'fruit': eaten += 1; fe += energy
            elif kind == 'predator': pd += 1; pen += energy/100; kills.append((round(t), round(age), round(energy)))
            else: sd += 1
        if os.environ.get('NIGHT_DEATHS'): deaths.extend([round(v, 1) if isinstance(v, float) else v for v in d] for d in eng.dbg_deaths())
    events()
    kw = dict(kw); pl = kw.pop('test_pred_life', 0.)
    eng.dbg_pred_life(float(pl))   # tests only: perfect-trap model (0 = off)
    eng.policy_init(nightsim.seed_key(seed), kw)
    peak = state['num_agents']; nxt = sample; last = None; traj = []; FD = {}; fates = []; tail = []
    if TAIL > 0: sample = 10.; nxt = 10.
    while True:
        info = eng.info(); n = len(eng.agents())
        if n == 0 or info['time'] >= horizon: break
        if DIAG_FROM > 0 and info['time'] >= DIAG_FROM - 1e-6:
            cp = fruit_diag(eng, horizon, nxt, FD, fates)
        else:
            _, cp = eng.run_policy(horizon, nxt)
        peak = max(peak, cp); events()
        info = eng.info()
        if info['time'] >= nxt - 1e-6:
            last = (len(eng.trees()), len(eng.fruits()), len(eng.agents()), len(eng.predators()))
            if TAIL > 0:
                ags = eng.agents()
                tail.append([int(nxt), len(ags), sum(1 for a in ags if a[4] > a[13]), round(sum(a[5] for a in ags)/max(1, len(ags))),
                             round(min((a[5] for a in ags), default=0)), sorted(round(a[4]) for a in ags), len(eng.trees()), eaten])
                tail = tail[-TAIL:]
            if abs(nxt % 250) < 1e-6:
                ags = eng.agents()
                # t, alive, trees, predators, fruit spawned so far (max id), eaten so far, eaten energy so far, mean energy
                tr = eng.dbg_trap(); held = 0
                if tr is not None:
                    import math as _m
                    held = sum(1 for p_ in eng.predators() if _m.hypot(p_[0] - tr[0], p_[1] - tr[1]) < 35)
                traj.append([int(nxt), last[2], last[0], last[3], info['next_fruit_id'], eaten, round(fe), round(sum(a[5] for a in ags)/max(1, len(ags))), round(sum(a[7] for a in ags)/max(1, len(ags)), 1), round(sum(a[10] for a in ags)/max(1, len(ags))), pd,
                             held, (tr[6] if tr else 0), (int(tr[4] >= 0) if tr else -1), (list(tr[8]) if tr else [0]*12)])
            nxt += sample
    info = eng.info()
    return dict(label=label, seed=seed, surv=round(info['time'], 1), score=round(info['score'], 3), fruit=round(fe/1000, 3),
                eaten=eaten, peak=peak, created=info['next_agent_id'], pdeaths=pd, sdeaths=sd, penalty=round(pen, 3),
                trees_d=last and last[0], fruits_d=last and last[1], preds=len(eng.predators()), traj=traj,
                wall=round(time.perf_counter()-t0, 1), **({'fates': fates} if DIAG_FROM > 0 else {}), **({'tail': tail} if TAIL > 0 else {}),
                **({'kills': kills} if os.environ.get('NIGHT_KILLS') else {}), **({'deaths': deaths} if os.environ.get('NIGHT_DEATHS') else {}), refuge=eng.dbg_eval(), corner=eng.corner_debug())

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', required=True); ap.add_argument('--seeds', nargs='+', required=True)
    ap.add_argument('--horizon', type=float, default=3000.); ap.add_argument('--predators', action='store_true')
    ap.add_argument('--workers', type=int, default=32); ap.add_argument('--out', required=True)
    ap.add_argument('--sample', type=float, default=50.)
    a = ap.parse_args()
    try: cfgs = json.loads(a.configs)
    except json.JSONDecodeError: cfgs = json.load(open(a.configs))
    if pathlib.Path(a.out).exists() and pathlib.Path(a.out).stat().st_size:
        ap.error('output already contains results; choose a new output for this revision')
    import nightsim, numpy
    sources=[HERE/'_nengine.cpp',HERE/'_npolicy.hpp',HERE.parent/'models/avoidance/native_corner.hpp']
    binary=pathlib.Path(nightsim._engine.__file__)
    if binary.stat().st_mtime < max(p.stat().st_mtime for p in sources):
        ap.error('native sources changed: rebuild with python nightsim/build.py')
    seeds = parse_seeds(a.seeds)
    manifest=dict(configs=cfgs,seeds=seeds,horizon=a.horizon,predators=a.predators,workers=a.workers,
        python=sys.version,platform=platform.platform(),numpy=numpy.__version__,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        sources={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.out).with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    jobs = [(l, kw, s, a.horizon, a.predators, a.sample) for s in seeds for l, kw in cfgs.items()]
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool, open(a.out, 'a') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush(); n += 1
    print(f'done {n} runs in {time.time()-t0:.0f}s -> {a.out}', flush=True)
