"""Replay leash scenarios captured during games (see run_game.py --scenarios): restore the engine
and the policy at the moment a leash started and run only that segment.

    ../.venv/bin/python scripts/trapper/replay_scenario.py results/trapper/scenarios/*.pkl.gz --seconds 40
"""
import argparse, glob, json, math, os, sys
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.utils.simulation import step_environment                        # noqa: E402
from models.trapper.scenario import restore                               # noqa: E402
from models.trapper.geometry import dist                                  # noqa: E402
from models.trapper.lure import predator_target                           # noqa: E402


def replay(path, seconds, verbose=False):
    env, policy, meta = restore(path)
    pid, guide = meta['pid'], meta['guide']
    # the observations the policy acts on come from the engine's current state
    state = dict(observations=[env.get_agent_state(a.agent_id) for a in env.agents], sim_time=env.time, num_agents=len(env.agents))
    t_end = env.time + seconds
    outcome = None; held_ticks = 0; last = ''
    while state['num_agents'] and state['sim_time'] < t_end:
        actions = policy(state['observations'], state['sim_time'])
        state = step_environment(env, actions, 0.1)
        m = policy.manager; w = policy.world
        d = m.deliveries.get(pid)
        if d is not None and d.guide == guide:
            last = d.decision
        for e in m.events[-6:]:
            if e['kind'] in ('delivered', 'delivery_failed', 'guide_captured') and e.get('pid') == pid and outcome is None and e['t'] >= meta['t']:
                outcome = (e['kind'], e.get('reason'), round(e['t'] - meta['t'], 1))
        if pid in m._held_prev:
            held_ticks += 1
        if verbose and int(round(state['sim_time'] * 10)) % 10 == 0 and w is not None:
            g = w.agents.get(guide); p = w.predator(pid)
            print(f"  t={state['sim_time']:6.1f} gap={dist(g.p, p.p) if g and p else -1:5.0f} ge={g.energy if g else -1:4.0f} pe={p.energy if p else -1:4.0f} rest={int(bool(p.resting)) if p else -1} tgt={predator_target(w, p) if p else None} | {last[:80]}")
    g_alive = any(a.agent_id == guide for a in env.agents)
    return dict(file=Path(path).name, outcome=outcome, held_ticks=held_ticks, guide_alive=g_alive, last=last[:90], meta=meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--seconds', type=float, default=40)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    files = [f for pat in a.files for f in sorted(glob.glob(pat))]
    ok = 0
    for f in files:
        r = replay(f, a.seconds, verbose=a.verbose)
        good = r['outcome'] is not None and r['outcome'][0] == 'delivered' or r['held_ticks'] >= 50
        ok += good
        print(f"{'OK  ' if good else 'FAIL'} {r['file']:52s} outcome={r['outcome']} held_ticks={r['held_ticks']} guide_alive={r['guide_alive']} | {r['last']}", flush=True)
    print(f'{ok}/{len(files)} scenarios delivered or held')
