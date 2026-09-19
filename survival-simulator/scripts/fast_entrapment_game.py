"""Run the ordinary policy on the verified C++ engine and record native-sprite replay chunks."""
import argparse, gzip, hashlib, json, math, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[1]

def atom(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,allow_nan=False)); tmp.replace(path)

def edges_from(obstacles):
    edges=set()
    for x,y,w,h in obstacles:
        edges.update((((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),((x+w,y+h),(x,y+h)),((x,y+h),(x,y))))
    return [[list(a),list(b)] for a,b in edges]

def world(env):
    agents=[]
    for a in env.agents:
        agents.append(dict(agent_id=a.agent_id,x=a.x,y=a.y,direction=a.direction,size=5,color=[128,128,128],
            energy=a.energy,max_energy=a.max_energy,age=a.age,hearing_radius=a.hearing_radius,
            vision_radius=a.vision_radius,cone_angle=a.cone_angle,_vision_poly=None))
    predators=[dict(x=p.x,y=p.y,direction=p.direction,size=10,color=[255,0,0],energy=p.energy,
        max_energy=200.,age=0.,hearing_radius=60.,vision_radius=250.,cone_angle=math.pi/3,_vision_poly=None)
        for p in env.predators]
    fruits=[dict(x=f.x,y=f.y,radius=f.radius,color=[0,255,0],age=f.age) for f in env.fruits]
    trees=[dict(x=t.x,y=t.y,radius=t.radius,color=[82,22,12],age=t.age) for t in env.trees]
    return dict(agents=agents,predators=predators,fruits=fruits,trees=trees)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,default=1883894846)
    p.add_argument('--seconds',type=float,default=3000); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bait-overlap',type=float,default=20.,help='Target overlap in seconds for bait replacement')
    p.add_argument('--bait-reserve',type=float,default=0.,help='Reserve a gathering donor this many seconds before estimated expiry')
    p.add_argument('--bait-food-lead',type=float,default=6.,help='Extra dispatch lead for a replacement that could eat en route')
    p.add_argument('--guide-lookahead',type=int,choices=(0,3),default=3,help='Future ticks to search, or 0 for legacy steering')
    p.add_argument('--guide-distance-min',type=float,default=100.,help='Preferred following distance lower bound')
    p.add_argument('--guide-distance-max',type=float,default=120.,help='Preferred following distance upper bound')
    p.add_argument('--no-shared-guide-paths',action='store_true',help='Disable guide forecast traffic avoidance for comparison')
    p.add_argument('--survival-config',type=Path,help='Optional JSON overrides for Orchard experiments')
    p.add_argument('--release-trap-food',action='store_true',help='Experimental separation of trap roles from Orchard workforce')
    p.add_argument('--nursery-size',type=int,default=0,help='Experimental number of gatherers producing bait near the rear entrance')
    p.add_argument('--summary-only',action='store_true',help='Record compact benchmark measurements without replay frames')
    default_fast = Path(os.environ.get('FASTSIM_ROOT', ROOT))
    p.add_argument('--fastsim',type=Path,default=default_fast,
                   help='Simulator source root containing fastsim/ (or set FASTSIM_ROOT)')
    a=p.parse_args(); folder=a.out; folder.mkdir(parents=True,exist_ok=True); (folder/'chunks').mkdir(exist_ok=True)
    if (folder/'summary.json').exists(): p.error('output already contains a run')
    sys.path[:0]=[str(a.fastsim.resolve()),str(ROOT)]
    import pygame
    from fastsim import SimulationCore
    from src.core import SimulationCore as PySimulationCore
    from models.core import EntrapmentPolicy
    # A separate Python initialization produces the identical static native background.
    py=PySimulationCore(seed=a.seed); bg=py.env.static_surface.copy(); bg.blit(py.env.shadow_surface,(0,0)); bg.blit(py.env.obstacle_surface,(0,0))
    pygame.image.save(bg,folder/'background.png'); del py
    settings={} if a.survival_config is None else json.loads(a.survival_config.read_text())
    sim=SimulationCore(seed=a.seed); env=sim.env; policy=EntrapmentPolicy(seed=a.seed,bait_overlap_seconds=a.bait_overlap,bait_reserve_seconds=a.bait_reserve,survival_settings=settings,release_trap_food=a.release_trap_food,nursery_size=a.nursery_size,bait_food_lead_seconds=a.bait_food_lead,guide_lookahead_ticks=a.guide_lookahead,share_guide_paths=not a.no_shared_guide_paths,guide_preferred_distance=(a.guide_distance_min,a.guide_distance_max)); started=time.monotonic()
    obstacles=[(o.x,o.y,o.width,o.height) for o in env.obstacles]; edges=edges_from(obstacles)
    atom(folder/'static.json',dict(width=env.width,height=env.height,edges=edges))
    sources=[*sorted((ROOT/'models').rglob('*.py')),*sorted((ROOT/'models').rglob('*.json')),Path(__file__),a.fastsim/'fastsim/_engine.cpp']
    atom(folder/'manifest.json',dict(seed=a.seed,horizon=a.seconds,bait_overlap_seconds=a.bait_overlap,bait_food_lead_seconds=a.bait_food_lead,guide_lookahead_ticks=a.guide_lookahead,guide_preferred_distance=[a.guide_distance_min,a.guide_distance_max],share_guide_paths=not a.no_shared_guide_paths,bait_reserve_seconds=a.bait_reserve,survival_overrides=settings,release_trap_food=a.release_trap_food,nursery_size=a.nursery_size,replay_frames=not a.summary_only,dt=sim.dt,engine='verified C++ fastsim',
        policy_inputs='Unmodified observations and simulation time only',sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}))
    states=sim.step([])['observations']; chunk=[]; history=[]; seen=set(); peak=0; first_bait=None; gap=longest=total_gap=0.; max_near=held30max=0; active={}; tick=0
    summary={}
    previous_states={}; previous_guide_plans={}; previous_actions={}
    sprint_deaths={}; premature_guide_deaths=[]; delivery_sacrifices=0
    death_counts={}; early_energy_deaths={}; fruit_count=ripe_count=0; energy_fraction_sum=energy_samples=0
    previous_roles={}
    try:
      for tick in range(round(a.seconds/sim.dt)+1):
        # Native evaluator events never enter the policy's observation inputs.
        native_events=[]
        for kind, when, aid, age, energy in sim.pop_events():
            role=previous_roles.get(aid,'unassigned')
            native_events.append(dict(kind=kind,time=when,agent=aid,age=age,energy=energy,role=role))
            if kind=='fruit':
                fruit_count+=1; ripe_count+=age>=20.
            else:
                key=kind+':'+role; death_counts[key]=death_counts.get(key,0)+1
                before=previous_states.get(aid)
                if kind=='predator' and before is not None:
                    can_sprint=before['energy']>=before['max_energy']/5
                    if can_sprint: sprint_deaths[role]=sprint_deaths.get(role,0)+1
                    if role=='guide':
                        plan=previous_guide_plans.get(aid,{})
                        intentional=plan.get('mode')=='hold_at_delivery'
                        delivery_sacrifices+=intentional
                        if can_sprint and not intentional:
                            premature_guide_deaths.append(dict(time=when,agent=aid,energy_before=before['energy'],
                                energy_at_death=energy,sprint_threshold=before['max_energy']/5,biome=before['biome'],
                                action=previous_actions.get(aid),guide_plan=plan))
                if kind=='starvation' and age<60.:
                    early_energy_deaths[role]=early_energy_deaths.get(role,0)+1
        energy_fraction_sum+=sum(s['energy']/s['max_energy'] for s in states);energy_samples+=len(states)
        now=env.time; terminal=not states or now>=a.seconds; actions=[] if terminal else policy(states,now); debug=policy.snapshot()
        holding=set(policy.retired_baits)|{policy.bait,policy.incoming}; baits=[x for x in env.agents if x.agent_id in holding and policy.site is not None and policy._arrival(x.agent_id)]
        if baits and first_bait is None:first_bait=now
        if first_bait is not None and not baits: gap+=sim.dt;total_gap+=sim.dt;longest=max(longest,gap)
        else:gap=0.
        near=0
        for i,pred in enumerate(env.predators):
            close=bool(baits) and min(math.hypot(pred.x-b.x,pred.y-b.y) for b in baits)<=40
            if close: near+=1;active.setdefault(i,now)
            else: active.pop(i,None)
        held30=sum(now-t>=30 for t in active.values());held30max=max(held30max,held30);max_near=max(max_near,near)
        seen.update(s['agent_id'] for s in states);peak=max(peak,len(states)); metrics=dict(agents=len(states),predators=len(env.predators),fruit=len(env.fruits),score=env.score,near_bait=near,held30=held30,bait_present_estimated=bool(baits),bait_energy=[b.energy for b in baits])
        if tick%10==0:history.append(dict(time=now,**metrics))
        if not a.summary_only:
            chunk.append(dict(tick=tick,time=now,world=world(env),policy=debug,input=states,actions=[x.model_dump() for _,x in actions],evaluation=metrics,native_events=native_events))
        if (tick+1)%100==0 or terminal:
            first=tick-len(chunk)+1
            if chunk:
                with gzip.open(folder/'chunks'/f'{first//100:05d}.json.gz','wt',compresslevel=1) as h:json.dump(chunk,h,separators=(',',':'),allow_nan=False)
            chunk=[];summary=dict(seed=a.seed,status='complete' if terminal else 'running',frames=tick+1,sim_time=now,runtime_seconds=time.monotonic()-started,score=env.score,final_agents=len(states),peak_agents=peak,total_agents_seen=len(seen),predators=len(env.predators),first_bait_time=first_bait,maximum_predators_within_40_of_bait=max_near,maximum_predators_continuously_near_bait_30s=held30max,estimated_bait_gap_seconds_after_first_arrival=total_gap,longest_estimated_bait_gap_seconds=longest,policy_metrics=policy.metrics,events=policy.events,history=history,metric_note='Proximity is a capture proxy; bait occupancy uses policy localization. C++ world state is evaluator/recorder only.')
            summary['native_evaluation']=dict(deaths_by_cause_and_role=death_counts.copy(),fruit_eaten=fruit_count,
                energy_deaths_before_age_60=early_energy_deaths.copy(),
                ripe_fruit_eaten=ripe_count,ripe_fraction=ripe_count/fruit_count if fruit_count else None,
                mean_agent_energy_fraction=energy_fraction_sum/energy_samples if energy_samples else None)
            summary['native_evaluation'].update(predator_deaths_with_sprint_available=sprint_deaths.copy(),
                premature_guide_predator_deaths_with_sprint_available=len(premature_guide_deaths),
                premature_guide_death_cases=premature_guide_deaths.copy(),intentional_delivery_sacrifices=delivery_sacrifices,
                sprint_benchmark_note='Availability at start of fatal tick. Intentional hold_at_delivery excluded from premature guide failures; terrain/walls can still make escape impossible.')
            atom(folder/'summary.json',summary)
        if terminal:break
        previous_roles=policy.roles.copy()
        previous_states={s['agent_id']:s for s in states}
        previous_actions={aid:action.model_dump() for aid,action in actions}
        previous_guide_plans={track.guide_id:track.memory.get('debug',{}).copy()
                              if isinstance(track.memory.get('debug'),dict) else {'mode':track.memory.get('debug')}
                              for track in policy.tracks.values() if track.guide_id is not None}
        states=sim.step(actions)['observations']
    except Exception:
        atom(folder/'error.json',dict(tick=tick,error=traceback.format_exc()));raise
    print(json.dumps({k:v for k,v in summary.items() if k not in ('history','events')},indent=2));print('Replay folder:',folder)

if __name__=='__main__':main()
