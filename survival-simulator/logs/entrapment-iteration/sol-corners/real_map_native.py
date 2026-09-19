"""Native 30+1 and rear-replacement trials for configuration-space pockets."""
from __future__ import annotations
import argparse, copy, gzip, heapq, json, math, os, random, sys
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest

def free(p,r,rects,w,h):
    x,y=p
    return r<=x<=w-r and r<=y<=h-r and not any(a-r<x<a+rw+r and b-r<y<b+rh+r for a,b,rw,rh in rects)

def clear(a,b,r,rects,w,h):
    n=max(1,math.ceil(math.dist(a,b)/2.0))
    return all(free((a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n),r,rects,w,h) for i in range(n+1))

def astar(start,goal,rects,w,h,step=5.):
    if clear(start,goal,5.01,rects,w,h): return [start,goal]
    origin=(min(start[0],goal[0])-100,min(start[1],goal[1])-100)
    def point(node): return origin[0]+node[0]*step,origin[1]+node[1]*step
    def node(p): return round((p[0]-origin[0])/step),round((p[1]-origin[1])/step)
    s=node(start); target=node(goal); queue=[(0.,s)]; cost={s:0.}; previous={}
    for _ in range(200000):
        if not queue: break
        _,cur=heapq.heappop(queue)
        if cur==target or math.dist(point(cur),goal)<step*1.5:
            path=[goal,point(cur)]
            while cur!=s: cur=previous[cur]; path.append(point(cur))
            path.append(start); path.reverse(); return path
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt=(cur[0]+dx,cur[1]+dy); q=point(nxt)
            if not free(q,5.01,rects,w,h) or not clear(point(cur),q,5.01,rects,w,h): continue
            nc=cost[cur]+math.hypot(dx,dy)*step
            if nc>=cost.get(nxt,1e99): continue
            cost[nxt]=nc; previous[nxt]=cur
            heapq.heappush(queue,(nc+math.dist(q,goal),nxt))
    raise RuntimeError('no agent path')

def command(agent,target,speed=5.):
    angle=math.atan2(target[1]-agent.y,target[0]-agent.x)-agent.direction
    return ActionRequest(agent_id=agent.agent_id,spawn_agent=False,move_distance=min(speed,math.dist((agent.x,agent.y),target)),move_direction=angle,turn_angle=angle)

def follows(agent,path,index):
    while index<len(path)-1 and math.dist((agent.x,agent.y),path[index])<3: index+=1
    return index,command(agent,path[index])

def sees(p,bait,env):
    return any(o.get('type')=='Agent' and o.get('id')==bait.agent_id for o in p.observe(agents=env._get_local_agents(p),edges=env._get_local_edges(p)))

def run(row,candidate,out):
    seed=row['seed']; rng=random.Random(seed^0xC0A7); core=SimulationCore(seed=seed,starting_agents=0,starting_predators=0); env=core.env
    rects=[(o.x,o.y,o.width,o.height) for o in env.obstacles]; goal=tuple(candidate['goal']); front=tuple(candidate['front']); rear=tuple(candidate['rear'])
    env.spawn_predator=lambda *a,**kw:None
    bait=Agent(*goal,rng=env.rng); bait.agent_id=0; bait.energy=bait.max_energy
    env.agents=[bait]; env.agents_dict={0:bait}; env._next_agent_id=1
    tracked=[]
    for _ in range(30):
        # Keep fixture variation tiny so every preload remains in hearing range.
        p=Predator(front[0]+rng.uniform(-.15,.15),front[1]+rng.uniform(-.15,.15),rng=env.rng); p.energy=p.max_energy;p.resting=False
        p.direction=math.atan2(goal[1]-p.y,goal[0]-p.x);tracked.append(p)
    env.predators=list(tracked);env._update_agent_grid();env._update_predator_grid()
    # Choose a predator-valid point 48--58 from bait, then route a guide from
    # 20 units along the static agent path. No live predator state informs it.
    newcomer_start=None; full_path=None
    for radius in (55.,52.,48.):
        for k in range(72):
            angle=2*math.pi*k/72; q=(goal[0]+radius*math.cos(angle),goal[1]+radius*math.sin(angle))
            if not free(q,10.01,rects,env.width,env.height): continue
            try: path=astar(q,goal,rects,env.width,env.height)
            except RuntimeError: continue
            newcomer_start=q;full_path=path;break
        if newcomer_start:break
    if newcomer_start is None: raise RuntimeError('no front delivery start')
    # Guide starts at the same route's first agent-valid point at least 20 away.
    guide_start=full_path[0]
    for p in full_path:
        if math.dist(p,newcomer_start)>=20: guide_start=p;break
    guide=Agent(*guide_start,rng=env.rng);guide.agent_id=1;guide.energy=guide.max_energy
    newcomer=Predator(*newcomer_start,rng=env.rng);newcomer.energy=newcomer.max_energy;newcomer.resting=False
    newcomer.direction=math.atan2(guide.y-newcomer.y,guide.x-newcomer.x)
    env.agents.append(guide);env.agents_dict[1]=guide;env.predators.append(newcomer);tracked.append(newcomer);env._update_agent_grid();env._update_predator_grid()
    guide_path=astar(guide_start,goal,rects,env.width,env.height);gi=1
    replacement=None;replacement_path=None;ri=1;arrived=False;old=bait;held_since=None;min_original=30;rear_intruders=set();rows=[];events=[];replacement_death=None
    native_kill=env.kill_agent
    def note_kill(agent):
        nonlocal replacement_death
        if replacement is not None and agent is replacement:
            contacts=[dict(id=i+1,distance=math.dist((p.x,p.y),(agent.x,agent.y)))
                      for i,p in enumerate(tracked) if math.dist((p.x,p.y),(agent.x,agent.y))<p.size+agent.size]
            replacement_death=dict(time=round(env.time+core.dt,6),position=[agent.x,agent.y],
                                   energy=agent.energy,age=agent.age,contacts=contacts,
                                   cause='predator_contact' if contacts and agent.energy>0 else 'energy_or_age')
        return native_kill(agent)
    env.kill_agent=note_kill
    replacement_tick=400; end_tick=800
    for tick in range(end_tick+1):
        now=round(tick*core.dt,6); distances=[math.dist((p.x,p.y),(bait.x,bait.y)) for p in tracked]
        held=[i for i,p in enumerate(tracked) if distances[i]<=40 and sees(p,bait,env)]
        min_original=min(min_original,sum(i<30 for i in held))
        if 30 in held: held_since=now if held_since is None else held_since
        else: held_since=None
        # Rear contact is the material replacement hazard. A half-plane test is
        # invalid for curved/multiwall pockets and produced false positives.
        for i,p in enumerate(tracked):
            if math.dist((p.x,p.y),rear)<15.05:rear_intruders.add(i+1)
        if tick==replacement_tick:
            replacement=Agent(*rear,rng=env.rng);replacement.agent_id=2;replacement.energy=replacement.max_energy
            env.agents.append(replacement);env.agents_dict[2]=replacement;env._update_agent_grid()
            stops=[rear]+[tuple(x) for x in candidate.get('replacement_via',[])]+[goal]
            replacement_path=[]
            for a,b in zip(stops,stops[1:]):
                leg=astar(a,b,rects,env.width,env.height)
                replacement_path.extend(leg if not replacement_path else leg[1:])
            events.append(dict(time=now,event='replacement started'))
        actions=[]
        if guide in env.agents and math.dist((guide.x,guide.y),goal)>1:
            gi,cmd=follows(guide,guide_path,gi);actions.append((1,cmd))
        if replacement is not None and replacement in env.agents and not arrived:
            if math.dist((replacement.x,replacement.y),goal)<=1:
                arrived=True;old.energy=0.;bait=replacement;events.append(dict(time=now,event='replacement arrived; old bait exhausted'))
            else:
                ri,cmd=follows(replacement,replacement_path,ri);actions.append((2,cmd))
        rows.append(dict(tick=tick,time=now,held=[i+1 for i in held],original_held=sum(i<30 for i in held),bait_id=bait.agent_id,
                         guide_alive=guide in env.agents,replacement_alive=replacement in env.agents if replacement else None,
                         replacement=None if replacement is None else [replacement.x,replacement.y,replacement.energy,replacement.age],
                         predators=[[p.x,p.y] for p in tracked]))
        if tick==end_tick:break
        bait.energy=bait.max_energy;bait.age=0.;core.step(actions);bait.energy=bait.max_energy;bait.age=0.
    result=dict(seed=seed,candidate=candidate,newcomer_start=list(newcomer_start),guide_start=list(guide_start),events=events,
                baseline_30_held=rows[0]['original_held']==30,minimum_original_held=min_original,
                newcomer_held_10s=held_since is not None and rows[-1]['time']-held_since>=10,final_held_count=len(rows[-1]['held']),
                rear_predator_intruders=sorted(rear_intruders),replacement_arrived=arrived,
                replacement_alive_at_end=replacement in env.agents if replacement else False,old_bait_retired=old not in env.agents)
    result['replacement_death']=replacement_death
    result['replacement_path']=replacement_path
    result['passed']=result['baseline_30_held'] and min_original==30 and result['newcomer_held_10s'] and result['final_held_count']==31 and not rear_intruders and arrived and result['replacement_alive_at_end'] and result['old_bait_retired']
    folder=out/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True);(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with gzip.open(folder/'ticks.jsonl.gz','wt') as f:
        for x in rows:f.write(json.dumps(x)+'\n')
    print(json.dumps({k:result[k] for k in ('seed','passed','baseline_30_held','minimum_original_held','newcomer_held_10s','final_held_count','rear_predator_intruders','replacement_arrived')}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--survey',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--goal',type=float,nargs=2);p.add_argument('--rear',type=float,nargs=2);p.add_argument('--via',type=float,nargs=2,action='append',default=[]);p.add_argument('seeds',type=int,nargs='+');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows={r['seed']:r for r in json.loads(a.survey.read_text())};results=[]
    for seed in a.seeds:
        candidate=copy.deepcopy(rows[seed]['candidates'][0])
        if a.goal:candidate['goal']=a.goal
        if a.rear:candidate['rear']=a.rear
        if a.via:candidate['replacement_via']=a.via
        results.append(run(rows[seed],candidate,a.output))
    (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
