"""Recorded evaluation for the observation-only temporary-holder intake gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import importlib.util
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WALL = HERE.parent / 'wall_funneling'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('wall_funnel_base_run', WALL/'run.py')
wall_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wall_base)
controlled_env = wall_base.controlled_env
add_predator = wall_base.add_predator
Obstacle = wall_base.Obstacle
ReplayRecorder = wall_base.ReplayRecorder
from predator_control import add_agent
from policy_pressure import IntakeGatePolicy
from src.utils.DTOs import ActionRequest, ObservationResponse

OUT = ROOT / 'results' / 'intake_gate_sol'
POLICY_HASH = hashlib.sha256((HERE/'policy_pressure.py').read_bytes()).hexdigest()


def run(predators=4, interval=90., seconds=420., seed=201, awake=False,
        native=False, spread=15., depth_spread=40.):
    assert predators >= 1 and interval > 0 and seconds > 0
    env = controlled_env()
    env.rng.seed(seed)
    rng = random.Random(seed)
    width, length, cx, cy = 30., 100., 800., 600.
    wall = Obstacle(cx-width/2, cy-length/2, width, length)
    env.obstacles = [Obstacle(0,0,1600,30), Obstacle(0,1170,1600,30),
        Obstacle(0,0,30,1200), Obstacle(1570,0,30,1200), wall]
    holders = []
    for offset in (-10., 10.):
        h = add_agent(env, x=cx+width/2+5.1, y=cy+offset, energy=150)
        h.direction = math.pi
        holders.append(h)
    env._next_agent_id = len(env.agents)
    env.edges = {tuple(sorted((a,b))) for o in env.obstacles for a,b in o.edges}
    env._update_spatial_grid()
    schedule = []
    for j in range(predators):
        schedule.append(dict(when=j*interval,
            px=cx-width/2-rng.uniform(190,190+depth_spread),
            py=cy+rng.uniform(-spread,spread),
            heading=rng.uniform(-.3,.3)))
    policy = IntakeGatePolicy(capacity=predators)
    tag = (f'intake-gate-sol-v3-n{predators}-i{interval:g}-s{seed}-'
           f'a{int(awake)}-{uuid4().hex[:8]}')
    replay = OUT/'replays'/f'{tag}.json.gz'
    rec = ReplayRecorder(env, title=tag, policy='intake-gate-sol-v3', seed=seed,
        every=10, scenario='30x100 wall; two far-face baits; individually scheduled predator/guide arrivals',
        notes=(__doc__ + ' Fixture supplies one guide at a prepared approach point per predator. '
               'All living agents receive unlimited food. Policy sees native DTO plus public time only.'),
        native_render=native, native_width=800, policy_sha256=POLICY_HASH)
    rec.capture()
    states = [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump()
              for a in env.agents]
    streaks, acquired, physical_losses, target_losses = {}, {}, {}, {}
    trace, deaths, tail = [], [], []
    spawned = 0
    for tick in range(round(seconds*10)):
        if Path('/tmp/predator-intake-stop').exists():
            rec.save(replay, reason='shared stop marker')
            raise RuntimeError('shared stop marker detected')
        due = [x for x in schedule if x['when'] <= env.time+.001]
        for item in due:
            aid = env._next_agent_id
            guide_x = cx-width/2-7 if spawned == 0 else cx-width/2-125
            guide = add_agent(env, x=guide_x, y=cy, energy=150)
            guide.agent_id = aid
            env._next_agent_id += 1
            env.agents_dict = {a.agent_id:a for a in env.agents}
            guide.direction = 0.
            p = add_predator(env, item['px'], item['py'], heading=item['heading'],
                             energy=102 if awake else 0)
            p.resting = not awake
            streaks[p] = 0
            spawned += 1
            schedule.remove(item)
        if due:
            states = [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump()
                      for a in env.agents]
        if not env.agents:
            break
        for a in env.agents:
            a.energy = a.max_energy
        inputs = json.loads(json.dumps(states))
        actions = policy.act(inputs, env.time)
        assert len(actions) == len(env.agents) == len({a['agent_id'] for a in actions})
        before = list(env.agents)
        pairs = []
        action_t = env.time
        for action in actions:
            a = env.agents_dict[action['agent_id']]
            assert 0 <= action['move_distance'] <= a.sprint_speed+.001
            assert all(math.isfinite(action[k]) for k in ('move_distance','move_direction','turn_angle'))
            req = ActionRequest(**action)
            pairs.append((a.agent_id, req))
            env.agent_step(**action)
        env.non_agent_step(.1)
        deaths.extend(dict(id=a.agent_id, time=round(env.time,1),
                           holder=a in holders)
                      for a in before if a.agent_id not in env.agents_dict)
        physical_count = target_count = 0
        for p in env.predators:
            seen = [o for o in p.observe(agents=list(env.agents), edges=list(env.edges))
                    if o['type'] == 'Agent']
            chosen = min(seen, key=lambda o:o['distance'])['id'] if seen else None
            physical = False
            targeted = False
            for h in holders:
                dx, dy = p.x-h.x, p.y-h.y
                candidate = (h.agent_id in env.agents_dict and math.hypot(dx,dy) <= 60
                    and p.x-h.x <= -(width+15.1)+.01
                    and abs(p.y-cy) < length/2-5)
                if candidate:
                    physical = True
                    targeted = p.resting or chosen == h.agent_id
                    if targeted:
                        break
            streaks[p] = streaks.get(p,0)+1 if physical else 0
            if streaks[p] >= 20 and p not in acquired:
                acquired[p] = round(env.time-1.9,1)
            if p in acquired and not physical and p not in physical_losses:
                physical_losses[p] = round(env.time,1)
            if p in acquired and not targeted and p not in target_losses:
                target_losses[p] = round(env.time,1)
            physical_count += int(physical)
            target_count += int(targeted)
        tail.append(physical_count)
        if tick % 10 == 0:
            st = next(iter(policy.stations.values()), None)
            trace.append(dict(time=round(env.time,1), spawned=spawned,
                physical=physical_count, targeted=target_count, alive=len(env.agents),
                observed_old=st['seen'] if st else 0,
                sleep_ready=st['sleep_ready'] if st else False))
        rec.capture(pairs, policy.decisions, inputs=states, action_t=action_t)
        states = [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump()
                  for a in env.agents]
    rec.save(replay, reason='horizon' if env.time >= seconds-.01 else 'all agents died')
    final30 = tail[-300:] if tail else [0]
    row = dict(policy='intake-gate-sol-v3', policy_hash=POLICY_HASH, seed=seed,
        predators=predators, interval=interval, requested_seconds=seconds,
        seconds=round(env.time,1), awake=awake, spawned=spawned,
        acquired=len(acquired), physical_losses=len(physical_losses),
        target_switches_after_acquisition=len(target_losses),
        max_physical=max(tail, default=0), final_physical=tail[-1] if tail else 0,
        tail30_min=min(final30), holders_alive=sum(h.agent_id in env.agents_dict for h in holders),
        success=(env.time >= seconds-.01 and spawned == predators and len(acquired) == predators
                 and not physical_losses and min(final30) == predators
                 and all(h.agent_id in env.agents_dict for h in holders)),
        deaths=deaths, events=policy.events, trace=trace,
        replay=str(replay.relative_to(ROOT)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{tag}.json').write_text(json.dumps(row, indent=2)+'\n')
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predators', type=int, default=4)
    p.add_argument('--interval', type=float, default=90)
    p.add_argument('--seconds', type=float, default=420)
    p.add_argument('--seed', type=int, default=201)
    p.add_argument('--awake', action='store_true')
    p.add_argument('--native', action='store_true')
    p.add_argument('--spread', type=float, default=15)
    p.add_argument('--depth-spread', type=float, default=40)
    result = run(**vars(p.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','trace')}, indent=2))
