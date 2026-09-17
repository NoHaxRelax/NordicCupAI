"""Paired original generated-map full games; no state leakage into controllers."""
from experiment import ROOT,OUT,SimulationCore,SimplePolicy,save,hashes,COMMIT,RunRecording,CONTROLLER_MODULE
from colony import CastePolicy
import argparse,json,time

def game(seed,mode,horizon=3000,*,reproduction=None,native=False,policy_class=None):
    start=time.perf_counter();sim=SimulationCore(seed=seed)
    policy=SimplePolicy('nursery',seed) if mode=='nursery' else (policy_class or CastePolicy)(seed,breeding=mode=='caste-breed')
    rec=RunRecording(sim.env,policy=mode,module=CONTROLLER_MODULE,seed=seed,
        scenario='generated',parameters=dict(seed=seed,mode=mode,horizon=horizon),
        horizon=horizon,reproduction=reproduction,native=native)
    state=sim.step([])
    rec.capture([],{},[],0.)
    samples=[];captures=0;births=5;assigned_active=0;assigned_target=0;preparation=[]
    while state['num_agents'] and state['sim_time']<horizon:
        inputs=state['observations'];action_t=state['sim_time']
        actions=policy(inputs,action_t)
        # Metrics observe actual pre-predator selection after actions are applied
        # in source order, then run the original non-agent step exactly once.
        before=list(sim.env.agents);roles=set(getattr(policy,'baits',{}));energy_before={a.agent_id:a.energy for a in before}
        for aid,act in actions:sim.env.agent_step(aid,act.move_distance,act.move_direction,act.turn_angle,act.spawn_agent)
        for p in sim.env.predators:
            if p.resting and p.energy<=100:continue
            seen=[o for o in p.observe(agents=sim.env._get_local_agents(p),edges=sim.env._get_local_edges(p)) if o['type']=='Agent']
            target=min(seen,key=lambda o:o['distance'])['id'] if seen else None
            if roles:assigned_active+=1;assigned_target+=target in roles
        sim.env.non_agent_step(.1)
        rec.capture(actions,getattr(policy,'last_decisions',{}),inputs,action_t)
        for a in before:
            if a not in sim.env.agents and a.energy>0:captures+=1
        state=dict(num_agents=len(sim.env.agents),sim_time=sim.env.time,score=sim.env.score,observations=[sim.env.get_agent_state(a.agent_id) for a in sim.env.agents])
        if round(state['sim_time']*10)%500==0:
            samples.append(dict(t=round(sim.env.time,2),alive=len(sim.env.agents),predators=len(sim.env.predators),baits=len(roles),births=sim.env._next_agent_id,score=sim.env.score))
    r=dict(seed=seed,mode=mode,horizon=horizon,duration=round(sim.env.time,2),score=round(sim.env.score,4),alive=len(sim.env.agents),births=sim.env._next_agent_id-5,captured_agents=captures,stats=getattr(policy,'stats',{}),active_predator_ticks_with_any_bait=assigned_active,target_fraction_when_any_bait=assigned_target/max(1,assigned_active),samples=samples,wall_seconds=time.perf_counter()-start)
    rec.finish(r,'extinction' if not sim.env.agents else 'requested horizon reached')
    save(f'fullgame-{mode}-{seed}',[r]);print(json.dumps(r),flush=True);return r
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=1);ap.add_argument('--mode',default='caste');ap.add_argument('--horizon',type=int,default=3000);a=ap.parse_args();game(a.seed,a.mode,a.horizon)
