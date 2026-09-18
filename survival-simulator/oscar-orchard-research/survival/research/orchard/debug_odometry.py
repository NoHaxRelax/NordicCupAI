"""Diagnostic only: compare the policy's dead-reckoned pose deltas with engine truth."""
import os, sys, pathlib, math, collections
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy, wrap

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 120
sim = SimulationCore(seed=seed); env = sim.env
pol = OrchardPolicy(seed=seed)
state = sim.step([])
prev_true = {}; prev_est = {}
bad = collections.Counter(); n = 0; shown = 0
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time'])
    # snapshot estimates and truth before the step
    est_before = {aid: (m.pose.p, m.pose.theta) for aid, m in pol.minds.items()}
    true_before = {a.agent_id: (a.x, a.y, a.direction) for a in env.agents}
    acts = {aid: a for aid, a in actions}
    state = sim.step(actions)
    true_after = {a.agent_id: (a.x, a.y, a.direction) for a in env.agents}
    # the policy's odometry runs at the start of the next call; emulate by calling _odometry on a copy
    for aid, m in list(pol.minds.items()):
        if aid not in true_before or aid not in true_after: continue
        import copy
        mm = copy.copy(m); mm.pose = copy.copy(m.pose)
        pol._odometry(mm, None)
        dx_est = (mm.pose.p[0]-est_before[aid][0][0], mm.pose.p[1]-est_before[aid][0][1])
        dth_est = wrap(mm.pose.theta-est_before[aid][1])
        # rotate estimate delta from group frame to world frame using heading offset
        off = wrap(true_before[aid][2]-est_before[aid][1])
        c, s = math.cos(off), math.sin(off)
        dx_w = (dx_est[0]*c-dx_est[1]*s, dx_est[0]*s+dx_est[1]*c)
        dx_true = (true_after[aid][0]-true_before[aid][0], true_after[aid][1]-true_before[aid][1])
        dth_true = wrap(true_after[aid][2]-true_before[aid][2])
        err = math.hypot(dx_w[0]-dx_true[0], dx_w[1]-dx_true[1]); n += 1
        if err > 0.5 or abs(dth_est-dth_true) > 1e-6:
            a = acts[aid]; ag = [x for x in env.agents if x.agent_id == aid][0]
            bad['pos' if err > 0.5 else 'heading'] += 1
            if shown < 12:
                shown += 1
                print(f"t={state['sim_time']:.1f} a{aid} err={err:.2f} est_d={math.hypot(*dx_w):.2f} true_d={math.hypot(*dx_true):.2f} "
                      f"act=(d={a.move_distance:.2f},dir={a.move_direction:.2f},turn={a.turn_angle:.2f}) biome={state['observations'] and next((o['biome'] for o in state['observations'] if o['agent_id']==aid), '?')} "
                      f"truepos=({true_after[aid][0]:.1f},{true_after[aid][1]:.1f}) e={ag.energy:.1f} sp={ag.speed:.1f}/{ag.sprint_speed:.1f} maxE={ag.max_energy:.0f}")
print('agent-ticks', n, 'bad', dict(bad), 'metrics', {k: v for k, v in pol.metrics.items() if k in ('pose_corrections', 'deflections', 'stuck_events')})
