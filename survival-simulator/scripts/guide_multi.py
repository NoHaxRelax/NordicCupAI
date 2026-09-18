"""Native one-map experiment: preload 30 at bait, then guide three sequentially.

The preload and arranged visible encounters are test fixtures. Policy inputs
stay observation-only; omniscient membership/retention metrics are evaluation.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import guide_lab as lab
from models.entrapment.my_guide import guide as policy


def replacement_side(point, site):
    """Conservative rear-mouth/hearing-zone occupancy, evaluation only."""
    axial=sum((point[k]-site['mouth'][k])*site['inward'][k] for k in range(2))
    return (axial > site['overlap']/2 and
            min(math.dist(point,site['goal']),math.dist(point,site['replacement_entry'])) <= 60.)


def write_viewer(folder):
    html=Path(lab.__file__).with_suffix('.html').read_text()
    html=html.replace('Cyan: guide · green: bait · yellow ring: handoff point · red: predator.',
        'Cyan: guide · green: bait · red: original predators · numbered rings: newcomers 31–33. Bottom-right: trap magnification. Overlapping predators are counted in the header.')
    html=html.replace('<p id="error">','<nav id="events"></nav><p id="error">')
    html=html.replace("run=r;", """run=r;
for(const d of r.deliveries||[]){
  for(const [label,time] of [[`Delivery ${d.number}`,d.start_time],[`Guide ${d.number} death`,d.capture?.time]]){
    if(time==null)continue;const b=document.createElement('button');b.textContent=label;
    b.onclick=()=>seek(Math.round(time/r.dt));$('events').appendChild(b);
  }
}
const b=document.createElement('button');b.textContent='Final state';b.onclick=()=>seek(r.frames-1);$('events').appendChild(b);
""")
    (folder/'index.html').write_text(html)


def run(args):
    started=time.monotonic()
    folder=args.output/f'multi-{args.seed}-{time.time_ns()}'
    folder.mkdir(parents=True)
    (folder/'frames').mkdir(); (folder/'ticks').mkdir()
    write_viewer(folder)
    shutil.copy2(__file__,folder/'multi_runner.py')
    for name in ('my_guide.py','guide_pathfinding.py','guide_steering.py','predator_following.py'):
        shutil.copy2(lab.ROOT/'models'/name,folder/('policy.py' if name=='my_guide.py' else name))
    core,site,site_count,bait,unused_guide,unused_predator=lab.setup(args.seed,args.encounter_seed,0)
    env=core.env
    geometry=lab._Geometry(env.width,env.height,[(o.x,o.y,o.width,o.height) for o in env.obstacles])
    env.agents=[bait]; env.agents_dict={0:bait}; env.predators=[]
    env.agent_observations.clear(); env._update_agent_grid()
    # Exactly 30+3: suppress only ambient additions, retaining all native steps.
    env.spawn_predator=lambda *a,**kw:None
    rng=random.Random(args.encounter_seed)
    tracked=[]
    for i in range(30):
        for _ in range(1000):
            # Use the validated approach lane, which may be offset from the
            # narrow gap's centreline. Native predators can share this space.
            p=tuple(site['handoff'][k]+rng.uniform(-.5,.5) for k in range(2))
            if geometry.free(p,10.01) and math.dist(p,site['goal'])>18.:break
        else: raise RuntimeError('No valid preload placement')
        predator=lab.Predator(*p,rng=env.rng)
        predator.direction=math.atan2(bait.y-p[1],bait.x-p[0])
        predator.energy=rng.uniform(80.,200.);predator.resting=False
        tracked.append(predator);env.predators.append(predator)
    env._update_predator_grid()
    summary=dict(seed=args.seed,encounter_seed=args.encounter_seed,site=site,eligible_sites=site_count,
                 experiment=f'30 preloaded + {args.deliveries} sequential deliveries',outcome='running',frames=0,
                 seconds=0.,dt=core.dt,deliveries=[],baseline={},
                 assumptions=['30 predators preplaced near the trap; first 30 deliveries are not tested',
                    'native simulator permits predator overlap; no predator separation force is added',
                    'new guide/predator encounters randomly placed at least 250 units from bait',
                    'guides start full; native energy/aging thereafter; survivors stand after their attempt',
                    'bait full energy and no aging; ambient predator additions disabled to isolate exactly 33',
                    'policy gets native observations and known map edges, never hidden predator state'],
                 success_rule='Newcomer near bait (<=40) and sensing it for 10s; all original predators retained; final 30s hold with replacement side clear',
                 policy_sha256=hashlib.sha256((lab.ROOT/'models/entrapment/my_guide.py').read_bytes()).hexdigest())
    lab.write_json(folder/'summary.json',summary)
    phase='settle';phase_start=0.;delivery=None;active=None;memory={};held_since=None;dead_since=None
    guides=[];events=[];departed=set();outside_hearing=set();min_initial=30;final_min=None;all_since=None
    stage_prior=set();stage_min=0;capture_events={};rear_final=set();rear_ever=set()
    native_kill=env.kill_agent
    def kill(agent):
        if agent in guides:
            touching=[i+1 for i,p in enumerate(tracked) if math.hypot(p.x-agent.x,p.y-agent.y)<p.size+agent.size]
            capture_events[agent.agent_id]=dict(time=round(env.time+core.dt,3),predators=touching,
                                             cause='predator' if touching and agent.energy>0 else 'energy_or_age')
        return native_kill(agent)
    env.kill_agent=kill

    def spawn_delivery(now):
        nonlocal active,memory,delivery,phase,phase_start,held_since,dead_since,stage_prior,stage_min
        number=len(summary['deliveries'])+1
        active=lab.Agent(0.,0.,rng=env.rng,color=(60,220,255));active.agent_id=number+1
        active.energy=active.max_energy
        predator=lab.Predator(0.,0.,rng=env.rng,color=(255,175,70));predator.energy=predator.max_energy;predator.resting=False
        for _ in range(10000):
            p=(rng.uniform(45,env.width-45),rng.uniform(45,env.height-45))
            angle=rng.uniform(-math.pi,math.pi);distance=rng.uniform(80,160)
            q=(p[0]+distance*math.cos(angle),p[1]+distance*math.sin(angle))
            if not geometry.free(p,5.01) or not geometry.free(q,10.01) or math.dist(p,site['goal'])<250:continue
            active.x,active.y=p;active.direction=angle;predator.x,predator.y=q;predator.direction=angle+math.pi
            seen=active.observe(agents=[bait],predators=[predator],edges=env.edges)
            if any(o['type']=='Predator' for o in seen):break
        else:raise RuntimeError('Could not sample visible encounter')
        env.agents.append(active);env.agents_dict[active.agent_id]=active;guides.append(active)
        env.predators.append(predator);tracked.append(predator)
        env._update_agent_grid();env._update_predator_grid()
        env.agent_observations[active.agent_id]=lab.observations(env,active)
        delivery=dict(number=number,predator_id=len(tracked),start_time=round(now,3),
                      guide_start=[active.x,active.y],predator_start=[predator.x,predator.y])
        summary['deliveries'].append(delivery)
        phase='delivery';phase_start=now;memory={};held_since=None;dead_since=None
        stage_prior=set(range(1,len(tracked)));stage_min=len(stage_prior)
        events.append(dict(time=round(now,3),event=f'Start delivery {number}: predator {len(tracked)}'))

    lab.pygame.init();screen=lab.pygame.Surface((640,480));font=lab.pygame.font.Font(None,22)
    with gzip.open(folder/'ticks.jsonl.gz','wt') as trace:
        for tick in range(2501):
            now=round(tick*core.dt,6);bait_alive=bait in env.agents
            held=[];distances=[]
            for i,p in enumerate(tracked):
                distance=math.hypot(p.x-bait.x,p.y-bait.y);distances.append(distance)
                seen=p.observe(agents=env._get_local_agents(p),edges=env._get_local_edges(p))
                if bait_alive and distance<=40 and any(o['type']=='Agent' and o.get('id')==bait.agent_id for o in seen):held.append(i+1)
            held_set=set(held)
            rear={i+1 for i,p in enumerate(tracked) if replacement_side((p.x,p.y),site)}
            rear_ever.update(rear)
            if phase!='settle':
                min_initial=min(min_initial,len(held_set&set(range(1,31))))
                departed.update(set(range(1,31))-held_set)
                outside_hearing.update(i+1 for i,d in enumerate(distances[:30]) if d>60.)
            if phase=='settle' and now>=10.:
                summary['baseline']=dict(held=len(held),passed=len(held)==30 and not rear,bait_alive=bait_alive,time=now,rear_ids=sorted(rear))
                if bait_alive and summary['baseline']['passed']:spawn_delivery(now)
            elif phase=='delivery':
                stage_min=min(stage_min,len(stage_prior&held_set))
                arrived=delivery['predator_id'] in held_set
                held_since=(now if held_since is None else held_since) if arrived else None
                passed=held_since is not None and now-held_since>=10.-1e-8
                alive=active in env.agents
                if not alive and dead_since is None:dead_since=now
                done=passed or now-phase_start>=60. or (dead_since is not None and now-dead_since>=20.)
                if done:
                    delivery.update(end_time=now,duration=round(now-phase_start,3),delivered=passed,
                                    guide_alive=alive,capture=capture_events.get(active.agent_id),
                                    total_held=len(held),prior_held_min=stage_min,
                                    prior_held_end=len(stage_prior&held_set))
                    events.append(dict(time=now,event=f'Delivery {delivery["number"]}: {"held" if passed else "failed"}; {len(held)}/{len(tracked)} held'))
                    print(events[-1],flush=True)
                    if len(summary['deliveries'])<args.deliveries and bait_alive:spawn_delivery(now)
                    else:
                        phase='final_hold';phase_start=now;active=None;final_min=len(held);all_since=None
            if phase=='final_hold':
                final_min=min(final_min,len(held))
                rear_final.update(rear)
                all_since=(now if all_since is None else all_since) if len(held)==30+args.deliveries else None
            terminal=not bait_alive or (phase=='settle' and now>=10.) or (phase=='final_hold' and now-phase_start>=30.-1e-8) or tick==2500
            inputs=action=None
            if active is not None and active in env.agents and not terminal:
                inputs=dict(bait=lab.local(site['goal'],active),edges=[[lab.local(a,active),lab.local(b,active)] for a,b in env.edges],
                            agent=copy.deepcopy(env.get_agent_state(active.agent_id)),
                            context=dict(tick=tick,dt=core.dt,time=now,handoff=lab.local(site['handoff'],active),mouth=lab.local(site['mouth'],active)))
                action=lab.validate_action(policy(**copy.deepcopy(inputs),memory=memory),active)
            evaluation=dict(phase=phase,total_predators=len(tracked),held_count=len(held),held_ids=held,
                            replacement_side_ids=sorted(rear),
                            initial_held=len(held_set&set(range(1,31))),bait_alive=bait_alive,
                            initial_ever_left_hold_zone=sorted(departed),initial_ever_beyond_60=sorted(outside_hearing),
                            active_guide_alive=active in env.agents if active else None,
                            all_33_continuous_seconds=0 if all_since is None else round(now-all_since,3),
                            predators=[dict(id=i+1,x=p.x,y=p.y,distance_to_bait=distances[i] if i<len(distances) else math.hypot(p.x-bait.x,p.y-bait.y),
                                            energy=p.energy,resting=p.resting) for i,p in enumerate(tracked)])
            event=next((e['event'] for e in reversed(events) if e['time']==now),'')
            row=dict(tick=tick,time=now,input=inputs,action=action.model_dump() if action else None,
                     debug=memory.get('debug'),evaluation=evaluation,event=event,error=None)
            trace.write(json.dumps(row)+'\n')
            if not args.bulk:
                lab.write_json(folder/'ticks'/f'{tick}.json',row)
                env.draw(screen)
                scale=640/env.width
                bx,by=round(bait.x*scale),round(bait.y*scale)
                crop=lab.pygame.Rect(bx-38,by-38,76,76).clamp(screen.get_rect())
                inset=lab.pygame.transform.scale(screen.subsurface(crop).copy(),(190,190))
                screen.blit(inset,(442,282));lab.pygame.draw.rect(screen,(235,240,245),(442,282,190,190),2)
                for i,p in enumerate(tracked[30:],31):
                    x,y=round(p.x*scale),round(p.y*scale)
                    lab.pygame.draw.circle(screen,(255,230,100),(x,y),7,1)
                    screen.blit(font.render(str(i),True,(255,240,160)),(x+8,y-10))
                lab.pygame.draw.rect(screen,(15,22,26),(0,0,640,27))
                screen.blit(font.render(f'{now:.1f}s | {phase} | held {len(held)}/{len(tracked)} | original {evaluation["initial_held"]}/30',True,(245,245,245)),(8,5))
                lab.pygame.image.save(screen,folder/'frames'/f'{tick}.png')
            summary.update(frames=tick+1,seconds=now,final=evaluation,events=events)
            if terminal:break
            bait.energy=bait.max_energy;bait.age=0.
            core.step([(active.agent_id,action)] if action else [])
            bait.energy=bait.max_energy;bait.age=0.
            if tick and tick%200==0:print(f'{now:.1f}s {phase}: held {len(held)}/{len(tracked)}',flush=True)
    count=30+args.deliveries
    outcome=('initial_trap_failed' if not summary['baseline'].get('passed') else
             'bait_dead' if not bait_alive else
             'wrong_side' if rear_final or rear else
             'initial_predators_escaped' if min_initial<30 else
             'delivery_pass' if len(held)==count and final_min==count and phase=='final_hold' and now-phase_start>=30.-1e-8 else
             'delivery_or_retention_failed')
    summary.update(outcome=outcome,replacement_side_final_period=sorted(rear_final),replacement_side_ever=sorted(rear_ever),
                   initial_min_held=min_initial,final_hold_min=final_min,
                   elapsed_seconds=round(time.monotonic()-started,3))
    lab.write_json(folder/'summary.json',summary)
    lab.pygame.quit()
    print(json.dumps({k:summary[k] for k in ('outcome','seconds','deliveries','baseline','initial_min_held','final_hold_min','elapsed_seconds')},indent=2),flush=True)
    print(f'Replay: http://127.0.0.1:9056/{folder.name}/index.html',flush=True)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=424736271)
    parser.add_argument('--encounter-seed',type=int,default=1335789813)
    parser.add_argument('--deliveries',type=int,default=3,choices=(1,3))
    parser.add_argument('--bulk',action='store_true')
    parser.add_argument('--output',type=Path,default=lab.ROOT/'logs/guide_lab')
    run(parser.parse_args())
