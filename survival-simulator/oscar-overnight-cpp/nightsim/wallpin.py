"""Wall-slide bait + pin test (tests only). An agent teleports into the corner (turn X; move_direction X), then slides north
inside the right wall with sprint moves (endpoint x beyond the map edge -> clamped back to W-5) until y ~= Y. Then NP awake
predators are placed at random spots 40-120 west of it (+-80 in y). The prisoner is kept alive (energy topped up). Rows:
seed, Y, NP, slide ok, final y, per-predator pinned time fraction and whether pinned at the end (within 60 of the bait)."""
import argparse, json, math, sys, pathlib, random, os
TRACK = float(os.environ.get('WP_TRACK', '0')); MODE = os.environ.get('WP_MODE', 'near')
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from nightsim.run import parse_seeds
X = 1e308
def A(aid, md=0., mdir=0., turn=0.): return (aid, dict(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=False))

def one(job):
    seed, Y, NP, hold = job
    import nightsim
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    sim.step([])
    k = eng.dbg_keep_agents(1); aid = k[0][0]
    eng.dbg_set_energy(aid, 480.)
    sim.step([A(aid, turn=X)]); sim.step([A(aid, mdir=X, turn=-X)])
    ag = [a for a in eng.agents() if a[0] == aid][0]
    if not (ag[1] == 1595. and ag[2] == 1195.): return dict(seed=seed, Y=Y, err='teleport', pos=(ag[1], ag[2]))
    ticks = 0; stuck = 0
    while ag[2] > Y + 1 and ticks < 200:
        L = min(20., ag[2] - Y + 10.5)
        dy = math.sqrt(max(0., L * L - 10.5 * 10.5))
        prev = ag[2]
        sim.step([A(aid, md=L, mdir=math.atan2(-dy, 10.5))]); ticks += 1
        eng.dbg_set_energy(aid, 480.)
        ag = [a for a in eng.agents() if a[0] == aid][0]
        if abs(ag[2] - prev) < 0.5: stuck += 1
        if stuck > 5: break
    out = dict(seed=seed, Y=Y, NP=NP, slide_ticks=ticks, x=round(ag[1], 2), y=round(ag[2], 1), slid=bool(abs(ag[2] - Y) < 25 and ag[1] == 1595.))
    if not out['slid']: return out
    rng = random.Random(seed * 100 + NP)
    placed = 0
    for _ in range(200):
        if placed >= NP: break
        px, py = ag[1] - rng.uniform(36, 52), ag[2] + rng.uniform(-30, 30)
        if eng.dbg_add_predator(px, py, rng.uniform(-3.14, 3.14), 150., False): placed += 1
    out['placed'] = placed
    inzone = [0] * placed
    T = int(hold * 10)
    track = TRACK
    for t in range(T):
        acts = []
        if track > 0.:
            st_ = sim.state()
            me = [o for o in st_['observations'] if o and o['agent_id'] == aid]
            if me:
                pr = [o for o in me[0]['observations'] if o['type'] == 'Predator']
                if pr:
                    dys = [o['distance'] * math.sin(o['angle']) for o in pr]   # heading is exactly 0 (east): +dy = south
                    if MODE == 'mid': dy = sum(dys) / len(dys)
                    elif MODE == 'far': dy = max(dys, key=abs)
                    elif MODE == 'span': dy = (max(dys) + min(dys)) / 2.
                    else:
                        q = min(pr, key=lambda o: o['distance']); dy = q['distance'] * math.sin(q['angle'])
                    if abs(dy) > track:
                        L = min(20., math.hypot(10.5, abs(dy)))
                        step = math.sqrt(max(0., L * L - 10.5 * 10.5))
                        acts = [A(aid, md=L, mdir=math.atan2(math.copysign(step, dy), 10.5))]
        sim.step(acts)
        if t % 50 == 0: eng.dbg_set_energy(aid, 480.)
        cur = [a for a in eng.agents() if a[0] == aid]
        if cur: ag = cur[0]
        prs = eng.predators()
        for i, p in enumerate(prs[:placed]):
            if math.hypot(p[0] - ag[1], p[1] - ag[2]) < 60.: inzone[i] += 1
        if not any(a[0] == aid for a in eng.agents()): out['bait_died'] = round(t / 10, 1); break
    prs = eng.predators()
    out['frac'] = [round(v / T, 2) for v in inzone]
    out['end_pinned'] = sum(1 for p in prs[:placed] if math.hypot(p[0] - ag[1], p[1] - ag[2]) < 60.)
    out['end_d'] = [round(math.hypot(p[0] - ag[1], p[1] - ag[2])) for p in prs[:placed]]
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', default=['1-40']); ap.add_argument('--out', required=True)
    ap.add_argument('--hold', type=float, default=300.); ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    jobs = [(s, Y, NP, a.hold) for s in parse_seeds(a.seeds) for Y in (1100, 900, 600, 300) for NP in (3, 5)]
    with Pool(a.workers) as pool, open(a.out, 'w') as f:
        for r in pool.imap_unordered(one, jobs):
            f.write(json.dumps(r) + '\n')
    print('done', flush=True)
