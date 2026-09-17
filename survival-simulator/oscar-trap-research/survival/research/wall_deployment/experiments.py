"""Bounded local evaluations; setup/evaluation privileges never enter WallPolicy.

Run from repo root with survival/.venv/bin/python .../experiments.py --help.
Generated mode preserves the complete native initial game and spawn rules.
Site mode arranges three founders and one awake predator at a natural wall;
terrain, obstacles, native food, native aging and all paid births remain intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor/survival-simulator'))
sys.path.insert(0, str(ROOT/'debugger'))
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest, ObservationResponse
from src.elements.predator import Predator
from controller import WallPolicy, add, sub, mul, dot

OUT = ROOT/'results/wall_deployment'


def verify_source():
    tree = json.loads((ROOT/'source-tree.json').read_text())
    files = []
    for entry in tree['tree']:
        prefix = 'survival-simulator/'
        if entry['type'] != 'blob' or not entry['path'].startswith(prefix): continue
        relative = entry['path'][len(prefix):]
        path = ROOT/'vendor/survival-simulator'/relative
        if not path.is_file(): continue
        data = path.read_bytes()
        sha = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        files.append(dict(path=relative, expected=entry['sha'], actual=sha, match=sha==entry['sha']))
    assert files and all(f['match'] for f in files), 'Vendored engine differs from pinned upstream'
    result = dict(commit=tree['sha'], checked=len(files), files=files)
    (OUT/'source-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def sites(env):
    candidates = []
    for wi, w in enumerate(env.obstacles[4:], 4):
        horizontal = w.height <= 35 and w.width >= 70
        if not horizontal and not (w.width <= 35 and w.height >= 70): continue
        thickness = w.height if horizontal else w.width
        length = w.width if horizontal else w.height
        center = (w.x+w.width/2, w.y+w.height/2)
        for side in (-1, 1):
            normal = (0, side) if horizontal else (side, 0)
            anchor = add(center, mul(normal, thickness/2+7))
            predator = add(center, mul(normal, -(thickness/2+12)))
            if env._in_obstacle(anchor, 5, env.obstacles) or env._in_obstacle(predator, 10, env.obstacles): continue
            def protected(p): return dot(sub(p, anchor), normal) >= 20 and math.dist(p, anchor) < 300
            trees = [t for t in env.trees if protected((t.x,t.y))]
            fruits = [f for f in env.fruits if protected((f.x,f.y))]
            # Deterministic pre-outcome natural patch ranking, a setup privilege.
            quality = sum(1-math.dist(anchor,(t.x,t.y))/400 for t in trees)*3+len(fruits)
            candidates.append(dict(wall_index=wi, wall=[w.x,w.y,w.width,w.height],
                anchor=anchor, predator=predator, normal=normal, thickness=thickness,
                length=length, trees=len(trees), fruits=len(fruits), quality=quality))
    return sorted(candidates, key=lambda s: (-s['quality'], s['wall_index'], s['normal']))


def prepare(seed, mode, native_predators, stress):
    env = SimulationCore(seed=seed).env
    site = None
    target_predator = None
    if mode != 'generated':
        choices = sites(env)
        if not choices: return None, None, None
        site = choices[0]
        anchor, normal = site['anchor'], site['normal']
        tangent = (-normal[1], normal[0])
        positions = [anchor]
        for offset in (20, -20):
            found = None
            for distance in (35, 40, 65, 90, 120):
                point = add(anchor, add(mul(normal, distance), mul(tangent, offset)))
                if not env._in_obstacle(point, 5, env.obstacles): found = point; break
            if found is None: return None, site, None
            positions.append(found)
        env.agents = env.agents[:3]
        env.agents_dict = {a.agent_id:a for a in env.agents}
        for a, p in zip(env.agents, positions):
            a.x, a.y = p
            a.direction = math.atan2(-normal[1], -normal[0])
        target_predator = Predator(*site['predator'], energy=102, rng=env.rng)
        target_predator.resting = False
        target_predator.direction = math.atan2(normal[1], normal[0])
        env.predators = [target_predator]
        if mode == 'rest':
            # Arranged opportunity, not a claimed natural encounter.
            target_predator.energy = 0
            target_predator.resting = True
            p = add(site['predator'], mul(normal, -28))
            if env._in_obstacle(p, 5, env.obstacles): return None, site, None
            env.agents[0].x, env.agents[0].y = p
            env.agents[0].direction = math.atan2(normal[1], normal[0])
        if stress:
            for distance in (200, 260, 320):
                p = add(anchor, mul(normal, distance))
                if not env._in_obstacle(p, 10, env.obstacles):
                    extra = Predator(*p, energy=102, rng=env.rng)
                    extra.resting = False
                    extra.direction = math.atan2(-normal[1], -normal[0])
                    env.predators.append(extra)
                    site['stress_predator'] = p
                    break
        env._update_spatial_grid()
    if not native_predators:
        env.spawn_predator = lambda *args, **kwargs: None
    return env, site, target_predator


def observe_initial(env):
    # Native initial DTOs are empty until the first world tick. Preserve this in
    # generated games. Site tests also start with empty DTOs, so no free sensing.
    return [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]


def run(seed=1, mode='site', wall=True, seconds=180, native_predators=True,
        stress=False, ripeness_wait=15., record=None, native_render=False, reproduction=False):
    env, site, target_predator = prepare(seed, mode, native_predators, stress)
    if env is None: return dict(seed=seed, mode=mode, skipped=True, site=site)
    policy = WallPolicy(wall=wall, ripeness_wait=ripeness_wait)
    births, deaths, meals, trace = [], [], [], []
    gen = {a.agent_id:0 for a in env.agents}
    native_spawn = env.spawn_agent
    def spawn_agent(*args, **kwargs):
        child = native_spawn(*args, **kwargs)
        parent = kwargs.get('parent')
        if parent is not None and child is not None:
            gen[child.agent_id] = gen[parent.agent_id]+1
            births.append(dict(time=round(env.time,1), parent=parent.agent_id, child=child.agent_id,
                               cost=100, child_energy=child.energy, generation=gen[child.agent_id]))
        return child
    env.spawn_agent = spawn_agent
    native_kill = env.kill_agent
    def kill_agent(a):
        deaths.append(dict(time=round(env.time,1), id=a.agent_id, age=round(a.age,1),
                           energy=round(a.energy,2), cause='depletion' if a.energy<=0 else 'capture'))
        native_kill(a)
    env.kill_agent = kill_agent
    native_remove = env.remove_fruit
    def remove_fruit(f):
        # Actual call's score increment distinguishes eaten old fruit from rot:
        # inspect proximity only for meal attribution, not controller decisions.
        eater = next((a for a in env.agents if math.dist((a.x,a.y),(f.x,f.y)) < a.size+f.radius), None)
        if eater is not None:
            meals.append(dict(time=round(env.time,1), agent=eater.agent_id, energy=round(f.energy,2)))
        native_remove(f)
    env.remove_fruit = remove_fruit
    recorder = None
    if record is None:
        record = OUT/'replays'/f'wall-v7-{mode}-seed{seed}-{"wall" if wall else "colony"}-{uuid4().hex[:10]}.json.gz'
    if record:
        from recorder import ReplayRecorder
        recorder = ReplayRecorder(env, title=f'Observation wall deployment / {mode} / seed {seed}',
            policy='wall_deployment.controller.WallPolicy', seed=seed, every=50,
            scenario='generated' if mode=='generated' else 'arranged natural site', native_render=native_render,native_width=640,
            notes='Policy receives ordinary DTOs only. Site setup and containment evaluation use true positions. '
                  'Native food, costs, births, aging; additional predator spawning '+str(native_predators)+'. '+
                  ('New reproduction of a previously unrecorded configuration; not original footage.' if reproduction else ''))
        recorder.capture()
    states = observe_initial(env)
    held = []
    registered_agents = 0
    pose_errors = []
    # Evaluation-only rigid transforms from each arbitrary local map to world.
    transforms = {}
    legal_actions = 0
    for tick in range(round(seconds*10)):
        if not env.agents: break
        action_t = env.time
        actions = policy.act(states, action_t)
        ids = [a['agent_id'] for a in actions]
        assert len(ids) == len(set(ids)) == len(states), 'Exactly one action per current agent'
        assert set(ids) == set(env.agents_dict), 'No newborn can act in its birth tick'
        for raw in actions:
            a = env.agents_dict[raw['agent_id']]
            assert 0 <= raw['move_distance'] <= a.sprint_speed+1e-6
            assert all(math.isfinite(raw[k]) for k in ('move_distance','move_direction','turn_angle'))
        # Reference transform is sampled before the first move for a new group.
        # Policy poses already contain its action prediction, so undo prediction.
        for raw in actions:
            aid = raw['agent_id']
            if aid not in policy.group_for: continue
            group = policy.group_for[aid]
            if id(group) not in transforms:
                a = env.agents_dict[aid]; predicted = policy.poses[aid]
                oldtheta = predicted.theta-raw['turn_angle']
                mod = env.biome_map[int(a.x),int(a.y)].move_penalty
                from controller import rot
                oldp = sub(predicted.p, rot((raw['move_distance']*mod,0), oldtheta+raw['move_direction']))
                rotation = a.direction-oldtheta
                transforms[id(group)] = (rotation, sub((a.x,a.y),rot(oldp,rotation)))
        pairs = [(a['agent_id'],ActionRequest(**a)) for a in actions]
        for aid, action in pairs:
            env.agent_step(aid, action.move_distance, action.move_direction, action.turn_angle, action.spawn_agent)
            legal_actions += 1
        env.non_agent_step(.1)
        env.agents_dict = {a.agent_id:a for a in env.agents}
        if site:
            normal, anchor = site['normal'], site['anchor']
            pred = (target_predator.x,target_predator.y)
            tangent = (-normal[1], normal[0])
            holds = (-(site['thickness']+55) < dot(sub(pred,anchor), normal) < -7 and
                     abs(dot(sub(pred,anchor),tangent)) < site['length']/2+10)
            held.append(holds)
        if tick%50 == 0:
            errors = []
            for a in env.agents:
                if a.agent_id not in policy.poses: continue
                from controller import rot
                rotation, origin = transforms[id(policy.group_for[a.agent_id])]
                estimate = add(origin, rot(policy.poses[a.agent_id].p,rotation))
                errors.append(math.dist(estimate,(a.x,a.y)))
            pose_errors.extend(errors)
            trace.append(dict(time=round(env.time,1), alive=len(env.agents), score=round(env.score,3),
                map_groups=len(policy.groups), map_edges=max((len(g.edges) for g in policy.groups),default=0),
                predators=len(env.predators), trees=len(env.trees), fruits=len(env.fruits),
                held=held[-1] if held else None, roles={str(k):v['rule'] for k,v in policy.decisions.items()},
                agents=[dict(id=a.agent_id,x=round(a.x,1),y=round(a.y,1),energy=round(a.energy,1),
                             age=round(a.age,1),generation=gen[a.agent_id]) for a in env.agents],
                max_pose_error=round(max(errors,default=0),3)))
        if recorder: recorder.capture(pairs,policy.decisions,inputs=states,action_t=action_t)
        states = [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]
    result = dict(seed=seed,mode=mode,wall=wall,horizon=seconds,seconds=round(env.time,1),
        native_predators=native_predators,stress=stress,ripeness_wait=ripeness_wait,
        site=site,alive=len(env.agents),score=round(env.score,4),births=births,deaths=deaths,meals=meals,
        captures=sum(d['cause']=='capture' for d in deaths),food_count=len(meals),
        food_energy=round(sum(m['energy'] for m in meals),2),max_generation=max(gen.values()),
        held_fraction=sum(held)/len(held) if held else None,
        first_hold_loss=next((round((i+1)*.1,1) for i,h in enumerate(held) if not h),None),
        held_final60=sum(held[-600:])/len(held[-600:]) if held else None,
        policy_metrics=policy.metrics,policy_events=policy.events,trace=trace,legal_actions=legal_actions,
        pose_error_mean=round(sum(pose_errors)/max(1,len(pose_errors)),4),
        pose_error_max=round(max(pose_errors,default=0),4),
        score_note='Native score: elapsed seconds plus gross fruit energy/1000 minus captured reserve/100.')
    if recorder:
        result['replay'] = str(record)
        recorder.save(record,reason='horizon' if env.agents else 'extinction')
    return result


def main():
    global WallPolicy
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',default='1,2')
    parser.add_argument('--mode',choices=['site','rest','generated'],default='site')
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--no-native-predators',action='store_true')
    parser.add_argument('--stress',action='store_true')
    parser.add_argument('--control',choices=['paired','wall','colony'],default='paired')
    parser.add_argument('--ripeness-wait',type=float,default=15.)
    parser.add_argument('--output',default='pilot.json')
    parser.add_argument('--record',action='store_true',help='Also capture original native rendering; every run always records states')
    parser.add_argument('--policy-file',type=Path,help='Replay an archived controller source version')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    source = verify_source()
    runs = []; start=time.perf_counter()
    policy_source = args.policy_file or Path(__file__).with_name('controller.py')
    if args.policy_file:
        import importlib.util
        spec=importlib.util.spec_from_file_location('wall_policy_snapshot',policy_source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        WallPolicy=module.WallPolicy
    policy_sha = hashlib.sha256(policy_source.read_bytes()).hexdigest()
    for seed in map(int,args.seeds.split(',')):
        for wall in ([True,False] if args.control=='paired' else [args.control=='wall']):
            record = OUT/'replays'/f'{Path(args.output).stem}-seed{seed}-{"wall" if wall else "colony"}-{uuid4().hex[:10]}.json.gz'
            row = run(seed=seed,mode=args.mode,wall=wall,seconds=args.seconds,
                native_predators=not args.no_native_predators,stress=args.stress,
                ripeness_wait=args.ripeness_wait,record=record,native_render=args.record)
            runs.append(row)
            print({k:row.get(k) for k in ('seed','mode','wall','seconds','alive','score','captures',
                        'food_count','held_fraction','first_hold_loss','policy_metrics')},flush=True)
            (OUT/args.output).write_text(json.dumps(dict(scope=__doc__,source_commit=source['commit'],
                policy_sha256=policy_sha,runs=runs),indent=2,default=lambda v:v.item())+'\n')
    print('runtime_seconds',round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__': main()
