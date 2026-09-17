"""Arranged wall/gap delivery test: one guide brings one predator to a prepared holder.

    ../.venv/bin/python scripts/trapper/test_delivery.py --kind wall --cases 6 --record

Outcome per case: whether the predator ends up held (its chosen target is the
holder and it sits in the front zone) for the final 15 s, whether the guide
survived, and the time of delivery.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.DTOs import ActionRequest                       # noqa: E402
from models.trapper.fixture import Arena                       # noqa: E402
from models.trapper.oracle import OracleWorld                  # noqa: E402
from models.trapper.geometry import Rect, add, mul, sub, dist, perp, unit   # noqa: E402
from models.trapper.motion import step_toward, speed_for   # noqa: E402
from models.trapper.sites import find_wall_sites, find_gap_sites      # noqa: E402
from models.trapper.lure import Lure, Holder, Delivery, predator_target   # noqa: E402
from models.trapper.recording import make_recorder, save_recorder     # noqa: E402


def run_case(kind, width, length, guide_offset, pred_angle, pred_distance, awake, seconds, record, native, seed, verbose, pred_heading=0.0):
    """guide_offset: (out, lateral) from the front face midpoint. The predator starts
    ``pred_distance`` from the guide at ``pred_angle`` (radians) measured from the
    site normal (0 = further out along the approach axis), heading toward the guide."""
    arena = Arena(seed=seed)
    env = arena.env
    if kind == 'wall':
        arena.add_rect(800 - width / 2, 600 - length / 2, width, length)          # vertical wall, front face toward -x
    else:
        arena.add_rect(800 - width / 2 - 60, 850 - length / 2, 60, length)         # two blocks, vertical passage, mouths top/bottom
        arena.add_rect(800 + width / 2, 850 - length / 2, 60, length + 10)
    oracle = OracleWorld(env)
    world = oracle.update()
    if kind == 'wall':
        sites = [s for s in find_wall_sites(world.rects, env.width, env.height) if s.normal[0] < 0]
    else:
        sites = [s for s in find_gap_sites(world.rects, env.width, env.height) if s.normal[1] < 0]
    assert sites, 'no site found in the fixture'
    site = sites[0]
    normal, axis = site.normal, site.axis
    holder_slot = site.holder
    guide_start = add(add(site.front_mid, mul(normal, guide_offset[0])), mul(axis, guide_offset[1]))
    ang = math.atan2(normal[1], normal[0]) + pred_angle
    pred_start = add(guide_start, (pred_distance * math.cos(ang), pred_distance * math.sin(ang)))
    holder = arena.add_agent(*holder_slot, heading=math.atan2(-normal[1], -normal[0]), energy=150) if kind == 'wall' else None
    guide = arena.add_agent(*guide_start, heading=0.0, energy=150)
    # the predator starts heading toward the guide (an encounter), offset by pred_heading
    heading = math.atan2(guide_start[1] - pred_start[1], guide_start[0] - pred_start[0]) + pred_heading
    pred = arena.add_predator(*pred_start, heading=heading, energy=102 if awake else 0.0, resting=not awake)
    rec = make_recorder(env, title=f'Trapper delivery test {kind} w{width} l{length} guide{guide_offset} angle{math.degrees(pred_angle):.0f} dist{pred_distance} {"awake" if awake else "resting"}',
                        policy='trapper lure v1 (oracle)', seed=seed, scenario='arranged flat forest; oracle world',
                        notes='guide lures one predator to a prepared holder; predator model exact', native=native, every=1) if record else None
    become_bait = kind == 'gap'
    delivery = Delivery(site=site, guide=guide.agent_id, pid=0, become_bait=become_bait, created=0.0)
    state = arena.step([])
    held_ticks = 0
    delivered_at = None
    trace = []
    ticks = int(seconds * 10)
    world = oracle.update(state['observations'])
    for t in range(ticks):
        lure, holder_ctl = Lure(world), Holder(world)
        actions = []
        decisions = {}
        for aid, a in world.agents.items():
            if holder is not None and aid == holder.agent_id:
                act, why = holder_ctl.act(a, holder_slot, site)
            elif aid == guide.agent_id and delivery.done is None:
                p = world.predator(delivery.pid)
                act = lure.act(delivery, a, p)
                why = delivery.decision
                if act is None:
                    # the lure defers to the society's flee; the fixture has none: run straight away at sprint
                    away = add(a.p, mul(unit(sub(a.p, p.p)), 40)) if p is not None else a.p
                    act = step_toward(a, away, speed_for(a, True))
                    why = 'fixture flee (' + why + ')'
            else:
                act, why = dict(agent_id=aid, move_distance=0., move_direction=0., turn_angle=0., spawn_agent=False), 'idle'
            actions.append((aid, ActionRequest(**act)))
            decisions[aid] = dict(rule=why)
        inputs = state['observations']
        action_t = state['sim_time']
        state = arena.step(actions)
        world = oracle.update(state['observations'])
        p = world.predator(0)
        target = predator_target(world, p) if p else None
        holder_ids = {holder.agent_id} if holder is not None else ({guide.agent_id} if delivery.done == 'delivered' else set())
        held = p is not None and target in holder_ids and site.in_front_zone(p.p, margin=10)
        if held:
            held_ticks += 1
            if delivered_at is None:
                delivered_at = round(state['sim_time'], 1)
        else:
            held_ticks = 0
        if rec:
            rec.capture(actions, decisions, inputs, action_t)
        if verbose and t % (1 if seconds <= 15 else 10) == 0:
            g = world.agents.get(guide.agent_id)
            trace.append(f"t={state['sim_time']:5.1f} phase={delivery.phase:8s} guide={('%.0f,%.0f' % g.p) if g else 'dead':9s} ge={g.energy if g else 0:.0f} pred={p.x:.0f},{p.y:.0f} e={p.energy:.0f} rest={p.resting} target={target} held={held} | {delivery.decision}")
        if guide.agent_id not in world.agents and delivery.done is None:
            delivery.done = 'guide_captured'
            delivery.event(state['sim_time'], 'guide_captured')
        if len(env.predators) > 1:
            pass
    if verbose:
        print('\n'.join(trace))
    final_hold = held_ticks >= 150
    result = dict(kind=kind, width=width, length=length, guide_offset=guide_offset, pred_angle_deg=round(math.degrees(pred_angle)), pred_distance=pred_distance,
                  awake=awake, guide_energy_left=round(env.agents_dict[guide.agent_id].energy, 1) if guide.agent_id in env.agents_dict else None, delivered_at=delivered_at, final_hold_s=round(held_ticks / 10, 1), success=bool(final_hold),
                  guide_alive=guide.agent_id in env.agents_dict, holder_alive=(holder.agent_id in env.agents_dict) if holder else None,
                  guide_outcome=delivery.done, events=delivery.events, extra_predators=len(env.predators) - 1)
    if rec:
        result['replay'] = str(save_recorder(rec, f'delivery-{kind}-w{int(width)}-l{int(length)}-a{round(math.degrees(pred_angle))}-d{int(pred_distance)}', reason='horizon'))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['wall', 'gap'], default='wall')
    ap.add_argument('--width', type=float, default=None)
    ap.add_argument('--length', type=float, default=100)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--cases', type=int, default=4)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    width = a.width if a.width is not None else (30.0 if a.kind == 'wall' else 15.0)
    configs = [
        dict(guide_offset=(320, 0), pred_angle=0.0, pred_distance=260, awake=True),          # ideal: predator further out on the axis
        dict(guide_offset=(320, 80), pred_angle=0.5, pred_distance=300, awake=True),
        dict(guide_offset=(300, -100), pred_angle=-0.8, pred_distance=280, awake=True),
        dict(guide_offset=(350, 150), pred_angle=1.4, pred_distance=300, awake=True),        # from the side
        dict(guide_offset=(350, -150), pred_angle=-1.6, pred_distance=260, awake=True),
        dict(guide_offset=(320, 0), pred_angle=0.3, pred_distance=280, awake=False),         # resting
        dict(guide_offset=(450, 60), pred_angle=2.6, pred_distance=300, awake=True),         # predator between guide and trap: needs a U-turn
        dict(guide_offset=(250, 200), pred_angle=0.9, pred_distance=180, awake=True),        # close encounter
    ][:a.cases]
    results = []
    for i, c in enumerate(configs):
        r = run_case(a.kind, width, a.length, seconds=a.seconds, record=a.record, native=a.native, seed=100 + i, verbose=a.verbose, **c)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'events'}), flush=True)
        print('   events:', r['events'])
    print(f'success {sum(r["success"] for r in results)}/{len(results)}; guides alive {sum(r["guide_alive"] for r in results)}')
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
