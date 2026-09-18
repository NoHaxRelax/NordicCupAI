"""Bait replacement on an arranged passage, driving the real TrapManager: a bait with little life
holds a predator at the mouth; a replacement is available some distance behind the far mouth.
Success: the predator is still held 20 s after the old bait dies.

    ../.venv/bin/python scripts/trapper/test_swap.py --cases 12
"""
import argparse, math, os, random, sys
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts' / 'trapper'))
from src.utils.DTOs import ActionRequest                                   # noqa: E402
from models.trapper.oracle import OracleWorld                              # noqa: E402
from models.trapper.geometry import add, mul, sub, dist                    # noqa: E402
from models.trapper.motion import hold                                     # noqa: E402
from models.trapper.sites import find_gap_sites                            # noqa: E402
from models.trapper.manager import TrapManager                             # noqa: E402
from models.trapper.lure import predator_target                            # noqa: E402
import test_leash as tl                                                    # noqa: E402


def run_case(seed, bait_energy, succ_dist, n_pred, seconds, verbose=False, **params):
    arena, rng = tl.build(seed, 4)
    env = arena.env
    oracle = OracleWorld(env); world = oracle.update()
    site = [s for s in find_gap_sites(world.rects, env.width, env.height) if s.normal[1] < 0 and abs(s.front_mid[0] - 800) < 1][0]
    mgr = TrapManager(**params)
    bait = arena.add_agent(*site.holder, heading=math.atan2(site.normal[1], site.normal[0]), energy=bait_energy)
    # predators charging the mouth from the front
    preds = []
    for k in range(n_pred):
        pp = add(site.front_mid, mul(site.normal, 40.0 + 25.0 * k))
        pp = (pp[0] + rng.uniform(-15, 15), pp[1])
        preds.append(arena.add_predator(*pp, heading=math.atan2(-site.normal[1], -site.normal[0]), energy=rng.uniform(60, 190)))
    # replacement candidate(s) behind the far mouth, plus a few idle agents far away (workers)
    back = mul(site.normal, -1.0)
    sp = add(site.far_mouth, mul(back, succ_dist))
    sp = (sp[0] + rng.uniform(-60, 60), sp[1])
    succ = arena.add_agent(*sp, heading=0.0, energy=400)
    idle = [arena.add_agent(200 + 100 * i, 150, heading=0.0, energy=300) for i in range(4)]
    state = arena.step([]); world = oracle.update(state['observations'])
    st = mgr._station(site)
    st.baits[bait.agent_id] = 0
    mgr.roles[bait.agent_id] = ('bait', st.key)
    st.staffed_since = 0.0
    # the bait is old: make the manager treat it as senescent (its life is its energy / drain)
    mgr.senescent.add(bait.agent_id)
    old_dead_at = None; held_after = 0; last_held = None; log = []
    for t in range(int(seconds * 10)):
        acts = mgr.step(world)
        actions = []
        for aid, a in world.agents.items():
            if aid in acts:
                act, why = acts[aid]
            else:
                act, why = hold(a), 'idle'
            actions.append((aid, ActionRequest(**act)))
        state = arena.step(actions); world = oracle.update(state['observations'])
        held = set(mgr._held_prev.keys())
        if bait.agent_id not in world.agents and old_dead_at is None:
            old_dead_at = round(state['sim_time'], 1)
        if old_dead_at is not None and state['sim_time'] > old_dead_at + 20.0:
            break
        if old_dead_at is not None:
            held_after += 1 if held else 0
        if verbose and t % 10 == 0:
            roles = {aid: r for aid, r in mgr.roles.items()}
            print(f"t={state['sim_time']:5.1f} held={sorted(held)} bait_e={world.agents[bait.agent_id].energy if bait.agent_id in world.agents else None} roles={roles} | {[acts[a][1][:40] for a in acts]}")
    ok = old_dead_at is not None and held_after >= 150 and bool(mgr._held_prev)
    ev = [e['kind'] for e in mgr.events if e['kind'] in ('successor_assigned', 'successor_in_place', 'bait_moved_up', 'bait_left', 'hold_ended', 'hold_started', 'bait_died')]
    return dict(seed=seed, ok=ok, old_dead_at=old_dead_at, held_after=held_after, final_held=sorted(mgr._held_prev.keys()), events=ev[-8:], succ_alive=succ.agent_id in world.agents)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=12); ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--bait-energy', type=float, default=40); ap.add_argument('--succ-dist', type=float, default=150)
    ap.add_argument('--preds', type=int, default=1); ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--verbose-seed', type=int, default=None); ap.add_argument('--reserve', type=int, default=1)
    a = ap.parse_args()
    params = dict(gap_reserve=bool(a.reserve), prestaff=False)
    if a.verbose_seed is not None:
        print(run_case(a.verbose_seed, a.bait_energy, a.succ_dist, a.preds, a.seconds, verbose=True, **params)); sys.exit(0)
    ok = 0
    for seed in range(a.start, a.start + a.cases):
        r = run_case(seed, a.bait_energy, a.succ_dist, a.preds, a.seconds, **params)
        ok += r['ok']
        print(f"seed {seed:3d}: {'OK  ' if r['ok'] else 'FAIL'} old bait died at {r['old_dead_at']}, held ticks after {r['held_after']}, final held {r['final_held']}, succ alive {r['succ_alive']} | {r['events']}", flush=True)
    print(f'success {ok}/{a.cases}')
