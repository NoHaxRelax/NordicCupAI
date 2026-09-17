"""Gap-trap endgame on an arranged passage: a bait already sits inside the mouth, a guide brings
one predator down the axis. Which guide behaviour hands the predator to the bait and survives?

    ../.venv/bin/python scripts/trapper/gap_endgame.py --seconds 40 --repeats 2
"""
import argparse, math, os, sys
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.utils.DTOs import ActionRequest                                   # noqa: E402
from models.trapper.fixture import Arena                                   # noqa: E402
from models.trapper.oracle import OracleWorld                              # noqa: E402
from models.trapper.geometry import add, mul, sub, dist, dot, unit, heading_of, wrap   # noqa: E402
from models.trapper.motion import step_toward, speed_for, hold, action     # noqa: E402
from models.trapper.sites import find_gap_sites                            # noqa: E402
from models.trapper.lure import Holder, predator_target                    # noqa: E402


def run(variant, depth, gap0, seed, width=14.0, length=60.0, seconds=40.0, stop_out=25.0, side_gap=40.0, energy=300.0):
    arena = Arena(seed=seed); env = arena.env
    arena.add_rect(800 - width / 2 - 60, 850 - length / 2, 60, length)
    arena.add_rect(800 + width / 2, 850 - length / 2, 60, length + 10)
    oracle = OracleWorld(env); world = oracle.update()
    site = [s for s in find_gap_sites(world.rects, env.width, env.height, depth=depth) if s.normal[1] < 0][0]
    normal, axis = site.normal, site.axis
    side = (-normal[1], normal[0])                       # along the obstacle front face
    bait = arena.add_agent(*site.holder, heading=math.atan2(normal[1], normal[0]), energy=200)
    g0 = add(site.front_mid, mul(normal, 100.0))
    guide = arena.add_agent(*g0, heading=math.atan2(-normal[1], -normal[0]), energy=energy)
    p0 = add(site.front_mid, mul(normal, 100.0 + gap0))
    pred = arena.add_predator(*p0, heading=math.atan2(g0[1] - p0[1], g0[0] - p0[0]), energy=150.0)
    state = arena.step([]); world = oracle.update(state['observations'])
    held_ticks = 0; first_hold = None; sidestepping = False; guide_dead_at = None
    for t in range(int(seconds * 10)):
        actions = []
        holder = Holder(world)
        for aid, a in world.agents.items():
            p = world.predator(0)
            if aid == bait.agent_id:
                act, _ = holder.act(a, site.holder, site)
                if dist(a.p, site.holder) < 0.8:
                    act = action(aid, 0.0, 0.0, wrap(math.atan2(normal[1], normal[0]) - a.heading))
            else:
                out = dot(sub(a.p, site.front_mid), normal)
                gap = dist(a.p, p.p) if p else 999
                if variant == 'sacrifice':
                    goal = add(site.front_mid, mul(normal, stop_out))
                    act = step_toward(a, goal, min(a.walk, dist(a.p, goal)), face=add(a.p, mul(normal, -50)))
                elif variant.startswith('sidestep'):
                    if not sidestepping and gap < side_gap and out < 60:
                        sidestepping = True
                    if sidestepping:
                        act = step_toward(a, add(a.p, mul(side, 40.0)), speed_for(a, True))
                    else:
                        goal = add(site.front_mid, mul(normal, stop_out))
                        act = step_toward(a, goal, min(a.walk, dist(a.p, goal)), face=add(a.p, mul(normal, -50)))
                elif variant == 'flyby':
                    if out > 14.0:
                        goal = add(site.front_mid, mul(normal, 12.0))
                        act = step_toward(a, goal, min(a.walk, dist(a.p, goal)), face=add(a.p, mul(normal, -50)))
                    else:
                        act = step_toward(a, add(a.p, mul(side, 40.0)), speed_for(a, True))
                else:
                    act = hold(a)
            actions.append((aid, ActionRequest(**act)))
        state = arena.step(actions); world = oracle.update(state['observations'])
        p = world.predator(0)
        target = predator_target(world, p) if p else None
        if guide.agent_id not in world.agents and guide_dead_at is None:
            guide_dead_at = round(state['sim_time'], 1)
        if bait.agent_id not in world.agents:
            return dict(variant=variant, depth=depth, gap0=gap0, seed=seed, bait_dead=round(state['sim_time'], 1), guide_dead=guide_dead_at, hold=0)
        if target == bait.agent_id and site.in_front_zone(p.p, margin=10):
            held_ticks += 1
            if first_hold is None:
                first_hold = round(state['sim_time'], 1)
    return dict(variant=variant, depth=depth, gap0=gap0, seed=seed, bait_dead=None, guide_dead=guide_dead_at,
                first_hold=first_hold, hold=round(held_ticks / (seconds * 10), 2), guide_energy=round(world.agents[guide.agent_id].energy) if guide.agent_id in world.agents else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seconds', type=float, default=40); ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--variants', default='sacrifice,sidestep30,sidestep40,sidestep55,flyby'); ap.add_argument('--depths', default='5,9.4')
    ap.add_argument('--width', type=float, default=14.0)
    a = ap.parse_args()
    for variant in a.variants.split(','):
        for depth in [float(x) for x in a.depths.split(',')]:
            for gap0 in (100, 125):
                rows = [run(variant, depth, gap0, seed, width=a.width, seconds=a.seconds,
                            side_gap=float(variant[8:]) if variant.startswith('sidestep') else 40.0) for seed in range(a.repeats)]
                print(f'{variant:10s} depth {depth:4.1f} gap0 {gap0}: ' + ' | '.join(
                    f"bait_dead={r['bait_dead']} guide_dead={r['guide_dead']} hold={r.get('hold')} first={r.get('first_hold')}" for r in rows), flush=True)
