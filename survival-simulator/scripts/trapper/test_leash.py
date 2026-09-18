"""Leash lead on arranged arenas with random obstacles: a guide with sprint energy brings a
predator (already chasing or not) to a gap mouth; optional second predator; staffed mouth (flyby)
or empty mouth (guide becomes the bait).

    ../.venv/bin/python scripts/trapper/test_leash.py --cases 24 --second 0 --seconds 60
"""
import argparse, math, os, random, sys
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.utils.DTOs import ActionRequest                                   # noqa: E402
from models.trapper.fixture import Arena                                   # noqa: E402
from models.trapper.oracle import OracleWorld                              # noqa: E402
from models.trapper.geometry import Rect, add, mul, sub, dist, free_point, path_clear   # noqa: E402
from models.trapper.motion import step_toward, speed_for, hold, action     # noqa: E402
from models.trapper.sites import find_gap_sites                            # noqa: E402
from models.trapper.lure import Lure, Holder, Delivery, predator_target    # noqa: E402


def build(seed, n_obstacles, width=14.0, length=60.0):
    rng = random.Random(seed)
    arena = Arena(seed=seed); env = arena.env
    arena.add_rect(800 - width / 2 - 60, 850 - length / 2, 60, length)
    arena.add_rect(800 + width / 2, 850 - length / 2, 60, length + 10)
    placed = []
    tries = 0
    while len(placed) < n_obstacles and tries < 400:
        tries += 1
        w, h = rng.uniform(30, 130), rng.uniform(30, 130)
        x, y = rng.uniform(60, 1500 - w), rng.uniform(60, 1100 - h)
        r = Rect(x, y, w, h)
        cx, cy = x + w / 2, y + h / 2
        # keep the corridor in front of the top mouth and the passage itself clear
        if abs(cx - 800) < 90 + w / 2 and 500 < cy < 950:
            continue
        if any(r.inflated(25).segment_hits((q.x, q.y), (q.x2, q.y2)) or q.inflated(25).contains((cx, cy)) for q in placed):
            continue
        placed.append(r)
        arena.add_rect(x, y, w, h)
    return arena, rng


def run_case(seed, n_obstacles, staffed, second, seconds, verbose=False, energy=500, nosprint=False, dmin=350, dmax=700):
    arena, rng = build(seed, n_obstacles)
    env = arena.env
    oracle = OracleWorld(env); world = oracle.update()
    site = [s for s in find_gap_sites(world.rects, env.width, env.height) if s.normal[1] < 0 and abs(s.front_mid[0] - 800) < 1][0]
    rects = world.rects
    def free(p, r=8.0):
        return free_point(p, r, rects, env.width, env.height)
    # guide somewhere in the open, 350-700 from the mouth
    for _ in range(500):
        g = (rng.uniform(80, 1520), rng.uniform(80, 760))
        if dmin <= dist(g, site.front_mid) <= dmax and free(g, 12):
            break
    for _ in range(500):
        ang = rng.uniform(0, 2 * math.pi); r = rng.uniform(120, 220)
        pp = add(g, (r * math.cos(ang), r * math.sin(ang)))
        if free(pp, 14) and 30 < pp[0] < 1570 and 30 < pp[1] < 1170:
            break
    guide = arena.add_agent(*g, heading=rng.uniform(0, 2 * math.pi), energy=energy, **({'sprint_speed': 10.0} if nosprint else {}))
    chasing = rng.random() < 0.6
    heading = math.atan2(g[1] - pp[1], g[0] - pp[0]) if chasing else rng.uniform(0, 2 * math.pi)
    pred = arena.add_predator(*pp, heading=heading, energy=rng.uniform(60, 190))
    bait = None
    if staffed:
        bait = arena.add_agent(*site.holder, heading=math.atan2(site.normal[1], site.normal[0]), energy=300)
    if second:
        for _ in range(500):
            ang = rng.uniform(0, 2 * math.pi); r = rng.uniform(200, 320)
            p2 = add(g, (r * math.cos(ang), r * math.sin(ang)))
            if free(p2, 14) and 30 < p2[0] < 1570 and 30 < p2[1] < 1170:
                break
        arena.add_predator(*p2, heading=rng.uniform(0, 2 * math.pi), energy=rng.uniform(60, 190))
    d = Delivery(site=site, guide=guide.agent_id, pid=0, become_bait=not staffed, created=0.0, leash=True)
    state = arena.step([]); world = oracle.update(state['observations'])
    held_ticks = 0; first_hold = None; ticks = int(seconds * 10); last = ''; e_done = None
    for t in range(ticks):
        lure = Lure(world, held=set(), baits={bait.agent_id} if bait else set())
        holder = Holder(world)
        actions = []
        for aid, a in world.agents.items():
            if bait is not None and aid == bait.agent_id:
                act, _ = holder.act(a, site.holder, site)
                if dist(a.p, site.holder) < 0.8:
                    act = action(aid, 0.0, 0.0, 0.0)
            elif aid == guide.agent_id:
                p = world.predator(d.pid)
                if d.done is None:
                    act = lure.act(d, a, p)
                    last = d.decision
                    if act is None:
                        act = hold(a)
                elif d.phase == 'FLYBY' and d.flee_heading is not None:
                    act = step_toward(a, add(a.p, mul(d.flee_heading, 40.0)), speed_for(a, True))
                else:
                    act = hold(a)
            else:
                act = hold(a)
            actions.append((aid, ActionRequest(**act)))
        state = arena.step(actions); world = oracle.update(state['observations'])
        p = world.predator(0)
        if p is None:
            break
        if e_done is None and d.done == 'delivered' and guide.agent_id in world.agents:
            e_done = world.agents[guide.agent_id].energy
        bait_ids = {bait.agent_id} if bait else ({guide.agent_id} if d.done == 'delivered' else set())
        # any predator held at this mouth counts (with a second predator the guide may lead that one)
        held = any(predator_target(world, q) in bait_ids and site.in_front_zone(q.p, margin=15) for q in world.predators)
        target = predator_target(world, p)
        if held:
            held_ticks += 1
            if first_hold is None:
                first_hold = round(state['sim_time'], 1)
        else:
            held_ticks = 0
        if guide.agent_id not in world.agents and (bait is None or d.done != 'delivered'):
            if bait is None:
                return dict(seed=seed, ok=False, why='guide died', t=round(state['sim_time'], 1), last=last, chasing=chasing)
        if verbose and t % 10 == 0:
            ga = world.agents.get(guide.agent_id)
            print(f"  t={state['sim_time']:5.1f} gap={dist(ga.p, p.p) if ga else -1:5.0f} pe={p.energy:4.0f} rest={p.resting} target={target} held={held} | {last}")
        if held_ticks >= 50:
            ga = world.agents.get(guide.agent_id)
            return dict(seed=seed, ok=True, t=first_hold, guide_alive=ga is not None, guide_energy=round(ga.energy) if ga else None, chasing=chasing, why='', dist0=round(dist(g, site.front_mid)), used=round(energy - (e_done if e_done is not None else (ga.energy if ga else energy))))
    ga = world.agents.get(guide.agent_id)
    return dict(seed=seed, ok=False, why='not held', t=None, last=last, chasing=chasing, guide_alive=ga is not None, phase=d.phase, done=d.done)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=24); ap.add_argument('--second', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=60); ap.add_argument('--obstacles', type=int, default=6)
    ap.add_argument('--staffed', type=int, default=-1, help='-1 alternate, 0 empty mouth, 1 bait present')
    ap.add_argument('--verbose-seed', type=int, default=None); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--energy', type=float, default=500); ap.add_argument('--nosprint', type=int, default=0)
    ap.add_argument('--dmin', type=float, default=350); ap.add_argument('--dmax', type=float, default=700)
    a = ap.parse_args()
    if a.verbose_seed is not None:
        r = run_case(a.verbose_seed, a.obstacles, a.staffed == 1 if a.staffed >= 0 else a.verbose_seed % 2 == 0, bool(a.second), a.seconds, verbose=True, energy=a.energy, nosprint=bool(a.nosprint))
        print(r); sys.exit(0)
    ok = 0; fails = []; results = []
    for seed in range(a.start, a.start + a.cases):
        staffed = (seed % 2 == 0) if a.staffed < 0 else bool(a.staffed)
        r = run_case(seed, a.obstacles, staffed, bool(a.second), a.seconds, energy=a.energy, nosprint=bool(a.nosprint), dmin=a.dmin, dmax=a.dmax)
        ok += r['ok']; results.append(r)
        print(f"seed {seed:3d} staffed={int(staffed)} chasing={int(r['chasing'])}: {'OK   ' if r['ok'] else 'FAIL '} t={r.get('t')} {r.get('why','')} {('e_left=' + str(r.get('guide_energy'))) if r['ok'] else ('| ' + r.get('last', ''))[:110]}", flush=True)
        if not r['ok']:
            fails.append(seed)
    print(f'success {ok}/{a.cases}; failed seeds {fails}')
    used = sorted(r['used'] for r in results if r.get('ok') and r.get('used') is not None)
    if used:
        print(f'energy used: mean {sum(used)/len(used):.0f}, median {used[len(used)//2]}, p90 {used[int(len(used)*0.9)]}, max {used[-1]}; '
              f'time to hold: median {sorted(r["t"] for r in results if r.get("ok"))[len(used)//2]}')
