"""Corner-pin scenario (tests only, engine truth): a prisoner sits in the teleport corner (1595,1195); a guide waits at the
corner's free tip; one awake predator approaches it from angle PHI (deg, direction FROM the guide to the predator) at
distance D. The guide arms when the predator is <= ARM away and teleports next tick. Pinned = predator still within
65 of the corner after HOLD s. Row: seed, phi, D, arm, eaten, pinned, final_d, min_d."""
import argparse, json, math, sys, pathlib
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
X = 1e308
CX, CY = 1595., 1195.

def A(aid, md=0., mdir=0., turn=0., spawn=False):
    return (aid, dict(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=spawn))

def one(job):
    seed, phi, D, arm, hold, gx, gy = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([])
    for (ox, oy, w, h) in eng.obstacles():
        if ox < 1570 and ox + w > 1440 and oy + h > 1040 and w < 1000 and h < 1000: return dict(seed=seed, skip=1)
    ks = eng.dbg_keep_agents(2)
    if len(ks) < 2: return None
    pa, ga = ks[0][0], ks[1][0]
    eng.dbg_set_energy(pa, 400.); eng.dbg_set_energy(ga, 400.)
    sim.step([A(pa, turn=X), A(ga)]); sim.step([A(pa, mdir=X, turn=-X), A(ga)])
    eng.dbg_teleport(ga, gx, gy, math.radians(phi))   # guide faces the predator
    b = math.radians(phi); px, py = gx + D * math.cos(b), gy + D * math.sin(b)
    if not eng.dbg_add_predator(px, py, b + math.pi, 200., False): return dict(seed=seed, phi=phi, D=D, skip=2)
    eng.pop_events()
    armed = False; tele = False; eaten = 0; mind = 1e9; t = 0
    while t < 400:
        ags = {a[0]: a for a in eng.agents()}; pr = eng.predators()[0]
        if ga not in ags: eaten = 1; break
        g = ags[ga]; d = math.hypot(pr[0] - g[1], pr[1] - g[2])
        if tele: break
        if armed: sim.step([A(pa), A(ga, mdir=X, turn=-X)]); tele = True; t += 1; continue
        if d <= arm: sim.step([A(pa), A(ga, turn=X)]); armed = True
        else: sim.step([A(pa), A(ga)])
        t += 1
    if eaten or not tele: return dict(seed=seed, phi=phi, D=D, arm=arm, eaten=eaten, pinned=0, tele=int(tele), gx=gx, gy=gy)
    fin = None
    for k in range(int(hold * 10)):
        ids = [a[0] for a in eng.agents()]
        sim.step([A(i) for i in ids])
        pr = eng.predators()[0]; dc = math.hypot(pr[0] - CX, pr[1] - CY); mind = min(mind, dc); fin = dc
        if not ids: break
    return dict(seed=seed, phi=phi, D=D, arm=arm, eaten=0, tele=1, pinned=int(fin is not None and fin < 65), final_d=round(fin or -1), min_d=round(mind), gx=gx, gy=gy)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', default=['1-60']); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--hold', type=float, default=60.)
    a = ap.parse_args()
    jobs = [(s, phi, D, arm, a.hold, gx, gy) for s in parse_seeds(a.seeds) for phi in (180, 195, 210, 225, 240, 255, 270) for D in (80, 150)
            for arm in (30, 36, 42, 50, 60) for (gx, gy) in ((1564.9, 1164.9), (1560., 1160.))]
    with Pool(a.workers) as pool, open(a.out, 'w') as f:
        for r in pool.imap_unordered(one, jobs, chunksize=8):
            if r: f.write(json.dumps(r) + '\n')
    print('done', flush=True)
