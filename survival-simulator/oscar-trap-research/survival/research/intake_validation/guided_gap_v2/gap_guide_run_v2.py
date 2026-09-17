"""Recorded observation-only guided intake into a size-selective gap.

The two obstacles, initial bait, and each off-axis guide/predator pair are
fixture supplied.  Recruitment and travel to those prepared starts are not
demonstrated.  Every agent action uses only its JSON-round-tripped native DTO
and public simulation time.  Hidden coordinates, targets, predator identities,
energy, and rest flags are used only by setup and post-action evaluation.
"""
from __future__ import annotations

import argparse,hashlib,json,math,random,sys
from pathlib import Path
from uuid import uuid4

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/"research"),str(ROOT/"debugger")]
from observed_gap_guide_policy_v2 import ObservedGapGuidePolicy
from predator_control import controlled_env,add_agent,add_predator
from recorder import ReplayRecorder,json_value
from src.elements.obstacle import Obstacle
from src.utils.DTOs import ActionRequest,ObservationResponse

OUT=ROOT/"results"/"intake_validation"/"guided_gap_v2"
POLICY_HASH=hashlib.sha256((HERE/"observed_gap_guide_policy_v2.py").read_bytes()+
                           (HERE/"observed_gap_base_v2_snapshot.py").read_bytes()).hexdigest()


def run(predators=4,interval=90.,seconds=400.,seed=4301,gap=15.,length=90.,
        right_length=70.,face_offset=-5.,horizontal=True,
        lateral=60.,approach=150.,follower_gap=65.,heading_jitter=.2,
        start_delay=6.,native=False,every=10):
    if Path("/tmp/predator-intake-stop").exists():raise SystemExit("stop sentinel present")
    rng=random.Random(seed);env=controlled_env();env.rng.seed(seed)
    cx,cy=800.,600.;base_line=500.;ow=60.
    overlap=min(length,face_offset+right_length)-max(0.,face_offset)
    if horizontal:
        left=Obstacle(base_line,cy-gap/2-ow,length,ow);right=Obstacle(base_line+face_offset,cy+gap/2,right_length,ow)
        mouth=(base_line+max(0.,face_offset),cy);inward=(1.,0.);cross=(0.,1.)
    else:
        left=Obstacle(cx-gap/2-ow,base_line,ow,length);right=Obstacle(cx+gap/2,base_line+face_offset,ow,right_length)
        mouth=(cx,base_line+max(0.,face_offset));inward=(0.,1.);cross=(1.,0.)
    env.obstacles=[Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200),left,right]
    env.edges={tuple(sorted(e)) for o in env.obstacles for e in o.edges}
    bait_start=(mouth[0]-inward[0]*15,mouth[1]-inward[1]*15)
    bait=add_agent(env,*bait_start,energy=150);bait.direction=math.atan2(inward[1],inward[0])
    env._next_agent_id=1;env._update_spatial_grid()
    policy=ObservedGapGuidePolicy();tag=f"observed-gap-guides-v2-g{gap:g}-l{length:g}x{right_length:g}-o{face_offset:g}-h{int(horizontal)}-n{predators}-i{interval:g}-lat{lateral:g}-s{seed}-{uuid4().hex[:8]}"
    replay=OUT/"replays"/f"{tag}.json.gz"
    rec=ReplayRecorder(env,title=tag,policy="observation-only-axial-gap-guides-v2",seed=seed,every=every,
        scenario="arranged size-selective gap and sequential off-axis guided predator arrivals",
        notes=__doc__+" Living agents are refilled before every tick; native predator movement, rest, sensing, collision, capture and target selection remain unchanged.",
        policy_sha256=POLICY_HASH,native_render=native,native_width=800)
    rec.capture();states=[ObservationResponse(**env.get_agent_state(bait.agent_id)).model_dump()]
    arrivals=[];streak={};physical_streak={};acquired={};physical_acquired={};losses={};physical_losses={}
    target_switch_ticks={};switch_run={};max_switch_run={};max_distance_after={}
    deaths=[];tail=[];physical_tail=[];trace=[];max_mapped=0;safe=set()
    for tick in range(round(seconds*10)):
        if Path("/tmp/predator-intake-stop").exists():break
        idx=len(arrivals);due=start_delay+idx*interval
        if idx<predators and env.time+.001>=due:
            side=-1 if idx%2 else 1
            gx=mouth[0]-inward[0]*approach+cross[0]*side*lateral
            gy=mouth[1]-inward[1]*approach+cross[1]*side*lateral
            vx,vy=gx-mouth[0],gy-mouth[1];scale=follower_gap/math.hypot(vx,vy)
            px,py=gx+vx*scale,gy+vy*scale
            aid=env._next_agent_id
            guide=add_agent(env,gx,gy,energy=150);guide.agent_id=aid;env._next_agent_id+=1
            env.agents_dict={a.agent_id:a for a in env.agents}
            guide.direction=math.atan2(mouth[1]-gy,mouth[0]-gx)
            predator=add_predator(env,px,py,heading=guide.direction+rng.uniform(-heading_jitter,heading_jitter),energy=102)
            arrivals.append(dict(predator=predator,guide=guide,scheduled=round(due,1),spawned=round(env.time,1),guide_start=[gx,gy],predator_start=[round(px,3),round(py,3)]))
            streak[predator]=physical_streak[predator]=0;target_switch_ticks[predator]=0
            switch_run[predator]=max_switch_run[predator]=0;max_distance_after[predator]=0.
            states=[ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]
        if not env.agents:break
        for a in env.agents:a.energy=a.max_energy
        inputs=json.loads(json.dumps(states));actions=policy.act(inputs,env.time)
        assert len(actions)==len(env.agents)==len({a["agent_id"] for a in actions})
        before=list(env.agents);pairs=[];action_t=env.time
        for action in actions:
            a=env.agents_dict[action["agent_id"]];assert 0<=action["move_distance"]<=a.sprint_speed+.001
            request=ActionRequest(**action);pairs.append((a.agent_id,request));env.agent_step(**action)
        env.non_agent_step(.1)
        deaths += [dict(id=a.agent_id,time=round(env.time,1),role="bait" if a is bait else "guide") for a in before if a.agent_id not in env.agents_dict]
        safe={a.agent_id for a in env.agents
              if 5<=((a.x-mouth[0])*inward[0]+(a.y-mouth[1])*inward[1])<=overlap-5
              and abs((a.x-mouth[0])*cross[0]+(a.y-mouth[1])*cross[1])<=gap/2-5+.01}
        joint=physical_count=0
        for item in arrivals:
            p=item["predator"];along=(p.x-mouth[0])*inward[0]+(p.y-mouth[1])*inward[1]
            physical=(math.dist((p.x,p.y),mouth)<=75 and along<=10)
            seen=[o for o in p.observe(agents=list(env.agents),edges=list(env.edges)) if o["type"]=="Agent"]
            chosen=min(seen,key=lambda o:o["distance"])["id"] if seen else None
            valid=physical and (p.resting or chosen in safe)
            physical_streak[p]=physical_streak[p]+1 if physical else 0;streak[p]=streak[p]+1 if valid else 0
            if physical_streak[p]>=20 and p not in physical_acquired:physical_acquired[p]=round(env.time-1.9,1)
            if streak[p]>=20 and p not in acquired:acquired[p]=round(env.time-1.9,1)
            if p in physical_acquired and not physical and p not in physical_losses:physical_losses[p]=round(env.time,1)
            if p in acquired and not valid and p not in losses:losses[p]=round(env.time,1)
            switched=p in acquired and not p.resting and chosen not in safe
            if switched:target_switch_ticks[p]+=1
            switch_run[p]=switch_run[p]+1 if switched else 0
            max_switch_run[p]=max(max_switch_run[p],switch_run[p])
            if p in physical_acquired:max_distance_after[p]=max(max_distance_after[p],math.dist((p.x,p.y),mouth))
            physical_count+=physical;joint+=valid
        tail.append(joint);physical_tail.append(physical_count);max_mapped=max(max_mapped,len(policy.stations))
        if tick%10==0:trace.append(dict(time=round(env.time,1),arrivals=len(arrivals),joint=joint,physical=physical_count,safe_agents=len(safe),alive=len(env.agents)))
        rec.capture(pairs,policy.decisions,inputs=states,action_t=action_t)
        states=[ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]
    reason="stop sentinel" if Path("/tmp/predator-intake-stop").exists() else "horizon" if env.time>=seconds-.01 else "all agents died"
    rec.save(replay,reason=reason);rows=[]
    for item in arrivals:
        p=item.pop("predator");g=item.pop("guide")
        rows.append(dict(**item,guide_id=g.agent_id,guide_alive=g.agent_id in env.agents_dict,acquired=acquired.get(p),physical_acquired=physical_acquired.get(p),first_joint_loss=losses.get(p),first_physical_loss=physical_losses.get(p),active_target_switch_ticks=target_switch_ticks[p],max_continuous_active_target_switch_seconds=round(max_switch_run[p]/10,1),max_distance_from_mouth_after_acquisition=round(max_distance_after[p],3)))
    n=min(300,len(tail));result=dict(policy_hash=POLICY_HASH,seed=seed,predators=predators,interval=interval,seconds=round(env.time,1),requested_seconds=seconds,gap=gap,length=length,right_length=right_length,face_offset=face_offset,horizontal=horizontal,overlap=overlap,lateral=lateral,approach=approach,follower_gap=follower_gap,arrivals=len(arrivals),mapped=max_mapped,agents_alive=len(env.agents),bait_alive=bait.agent_id in env.agents_dict,refuge_agents_alive=len(safe),guides_alive=sum(r["guide_alive"] for r in rows),deaths=deaths,acquired=len(acquired),physical_acquired=len(physical_acquired),joint_losses=len(losses),physical_losses=len(physical_losses),max_distance_from_mouth_after_acquisition=round(max(max_distance_after.values(),default=0.),3),max_continuous_active_target_switch_seconds=round(max(max_switch_run.values(),default=0)/10,1),final_joint=tail[-1],final_physical=physical_tail[-1],tail_joint_min=min(tail[-n:]),tail_physical_min=min(physical_tail[-n:]),joint_success=len(arrivals)==predators and len(acquired)==predators and not losses and env.time>=seconds-.01 and min(tail[-n:])==predators,physical_success=len(arrivals)==predators and len(physical_acquired)==predators and not physical_losses and env.time>=seconds-.01 and min(physical_tail[-n:])==predators,deliveries=rows,trace=trace,replay=str(replay.relative_to(ROOT)))
    result=json_value(result)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/f"{tag}.json").write_text(json.dumps(result,indent=2)+"\n");return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [("predators",4),("interval",90.),("seconds",400.),("seed",4301),("gap",15.),("length",90.),("right-length",70.),("face-offset",-5.),("lateral",60.),("approach",150.),("follower-gap",65.),("heading-jitter",.2),("start-delay",6.),("every",10)]:
        p.add_argument("--"+name,type=int if isinstance(default,int) else float,default=default)
    p.add_argument("--horizontal",action="store_true",default=True);p.add_argument("--vertical",dest="horizontal",action="store_false")
    p.add_argument("--native",action="store_true");r=run(**vars(p.parse_args()))
    print(json.dumps({k:v for k,v in r.items() if k not in ("trace","deliveries")},indent=2))
