"""Exact fork checkpoints; twenty families, 32 trials on 200 common late states."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pathlib,sys,time,traceback,hashlib
import multiprocessing as mp
import numpy as np
import tune_families10 as bo
ROOT=bo.ROOT
BASE=json.loads((ROOT/'docs/families20/scheduled_breeding/winner.json').read_text())['config']
FAMILIES=[
 ('late_reserves',{}, {'breed_reserve_late':(105,330),'low_pop_reserve':(105,240),'emergency_reserve':(80,180)}),
 ('small_population',{}, {'cap_min':(1,8),'cap_hard_min':(1,5),'cap_mult':(.1,.8),'tree_half':(250,1200)}),
 ('tree_capacity',{}, {'tree_slots':(1,3),'cap_tree_slack':(0,8),'cap_max':(10,65),'cap_mult':(.2,1.2)}),
 ('early_heirs',{}, {'heir_age':(30,75),'heir_reserve':(120,330),'heir_slack':(0,.5),'low_pop_reserve':(105,220)}),
 ('elite_heirs',{}, {'fit_vision':(.1,2),'fit_energy':(0,2),'fit_speed':(0,2),'fit_hear':(0,1.5),'heir_slack':(0,.4)}),
 ('nursery',{'nursery_bonus':20.}, {'nursery_bonus':(0,150),'heir_age':(35,80),'heir_reserve':(120,300),'old_reach':(30,160)}),
 ('breed_feeding',{'feed_mode':'breed'}, {'breed_reserve_late':(110,300),'heir_reserve':(120,300),'fruit_reach':(60,300),'ripen_wait':(5,32)}),
 ('old_priority',{'old_eat_last':False}, {'old_reach':(30,220),'heir_age':(35,85),'heir_reserve':(120,300),'rot_margin':(32,49)}),
 ('retirement',{}, {'no_eat_age':(55,130),'old_reach':(15,140),'heir_age':(35,80),'heir_reserve':(120,280)}),
 ('ripe_food',{}, {'ripen_wait':(5,38),'rot_margin':(28,49),'hungry_margin':(0,60),'fruit_min_wait':(0,5)}),
 ('local_food',{}, {'fruit_reach':(35,220),'tree_reach':(100,500),'dist_pen':(.05,1),'post_radius':(10,70)}),
 ('wide_food',{}, {'fruit_reach':(180,500),'tree_reach':(350,850),'lone_reach_mult':(1,3),'explore_energy':(70,240)}),
 ('cluster_food',{'cluster_radius':70.}, {'cluster_radius':(15,180),'tree_slots':(1,3),'dist_pen':(.05,.8),'post_radius':(10,70)}),
 ('spread_food',{'spread_weight':.5}, {'spread_weight':(.05,2),'tree_reach':(200,750),'repost_every':(2,25),'switch_gain':(10,150)}),
 ('relocate',{}, {'switch_gain':(0,160),'min_stay':(0,50),'repost_every':(1,25),'watch_refresh':(5,90)}),
 ('conserve_explore',{}, {'explore_energy':(80,320),'watch_patience':(10,120),'watch_reach':(150,700),'explore_radius':(100,700)}),
 ('scan',{}, {'sweep_rate':(.005,.2),'travel_turn':(.05,.8),'watch_refresh':(5,100),'watch_patience':(10,100)}),
 ('birth_pacing',{}, {'births_per_tick':(1,5),'breed_reserve_late':(110,330),'low_pop_reserve':(105,250),'cap_min':(1,8)}),
 ('late_gaze',{'pred_gaze':1}, {'pred_r':(60,160),'pred_sprint_r':(25,110),'pred_dodge_ang':(.5,1.8),'pred_turn_max':(.2,1.5)}),
 ('economy_mix',{}, {'breed_reserve_late':(110,300),'ripen_wait':(8,32),'fruit_reach':(60,300),'explore_energy':(80,280),'heir_reserve':(120,300),'dist_pen':(.05,.8)}),
]
INTEGER={'cap_min','cap_hard_min','cap_max','tree_slots','cap_tree_slack','births_per_tick'}
TRAIN=list(range(11001,11201))

def configuration(i,x):
    _,fixed,ranges=FAMILIES[i];c={**BASE,**fixed}
    for (k,(lo,hi)),v in zip(ranges.items(),x):
        value=float(lo+(hi-lo)*v);c[k]=int(round(value))if k in INTEGER else value
    if i==18:c['pred_dodge_r']=c['pred_r']
    return c

def initial(i):
    c={**BASE,**FAMILIES[i][1]}
    return np.array([np.clip(((c[k]if np.isfinite(c[k])else hi)-lo)/(hi-lo),0,1)for k,(lo,hi)in FAMILIES[i][2].items()])

def continuation(sim,cfg,meta):
    sim._engine.policy_init([0],cfg,True)
    start=time.process_time_ns();steps,peak,iface,policy,engine=sim._engine.run_policy(3000.,3000.,True)
    return dict(seed=meta['seed'],score=sim.env.score,gain=sim.env.score-meta['score'],survival=sim.env.time,
                steps=steps,ns_cpu=time.process_time_ns()-start,ns_policy=policy,ns_engine=engine)

def fork_run(sim,cfg,meta):
    read,write=os.pipe();pid=os.fork()
    if pid==0:
        os.close(read)
        try:result={'result':continuation(sim,cfg,meta)}
        except BaseException:result={'error':traceback.format_exc()}
        with os.fdopen(write,'w')as f:json.dump(result,f)
        os._exit(0)
    os.close(write)
    with os.fdopen(read)as f:result=json.load(f)
    _,status=os.waitpid(pid,0)
    if status or 'error'in result:raise RuntimeError(result)
    return result['result']

def actor(conn,seeds):
    try:
        from fastsim.fastpolicy import PolicySimulationCore
        saved=[];metadata=[]
        for seed in seeds:
            baseline=PolicySimulationCore(seed=seed,predators=True);baseline.policy_init(0,BASE)
            steps,*_=baseline.run_policy(3000.);end=baseline.env.time;score=baseline.env.score
            if len(baseline.env.agents):raise RuntimeError(f'Seed {seed} survived horizon; cannot label it a pre-extinction checkpoint')
            tick=max(0,steps-2500)
            if tick==0:raise RuntimeError(f'Seed {seed} died before 250 seconds')
            del baseline
            sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,BASE)
            sim.run_policy(3000.,tick*.1)
            meta=dict(seed=seed,time=sim.env.time,score=sim.env.score,baseline_end=end,baseline_score=score,checkpoint_tick=tick)
            resumed=fork_run(sim,BASE,meta)
            if resumed['score']!=score or resumed['survival']!=end:raise RuntimeError(f'Checkpoint replay mismatch: {meta} {resumed}')
            saved.append(sim);metadata.append(meta)
        conn.send({'ready':metadata})
        while True:
            cfg=conn.recv()
            if cfg is None:break
            conn.send({'rows':[fork_run(sim,cfg,meta)for sim,meta in zip(saved,metadata)]})
    except BaseException:conn.send({'error':traceback.format_exc()})
    finally:conn.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pod-index',type=int,required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=32);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();assert 0<=a.pod_index<10
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Use a new run directory')
    sources=[pathlib.Path(__file__),pathlib.Path(sys.argv[0]).resolve(),ROOT/'scripts/tune_families10.py']+list((ROOT/'fastsim').glob('*.cpp'))+list((ROOT/'fastsim').glob('*.hpp'))
    manifest=dict(source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},baseline=BASE,
      train_seeds=TRAIN,iterations=32,objective='mean score gained after checkpoint',window_seconds=250,policy_seed=0,
      families=FAMILIES,checkpoint='Linux fork copy-on-write process state, including native engine and policy RNG/memory',
      build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text()),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
    save('manifest.json',manifest);ctx=mp.get_context('fork');actors=[];pipes=[];start=time.monotonic()
    seeds=TRAIN[:2]if a.smoke else TRAIN;workers=min(a.workers,len(seeds))
    try:
        for k in range(workers):
            parent,child=ctx.Pipe();p=ctx.Process(target=actor,args=(child,seeds[k::workers]));p.start();child.close();actors.append(p);pipes.append(parent)
        def receive():
            result=[]
            for c in pipes:
                if not c.poll(3600):raise TimeoutError('Checkpoint actor timeout')
                r=c.recv()
                if 'error'in r:raise RuntimeError(r['error'])
                result.append(r)
            return result
        metas=[m for r in receive()for m in r['ready']];save('checkpoints.json',sorted(metas,key=lambda x:x['seed']))
        print(json.dumps(dict(checkpoints=len(metas),verified=True,elapsed=time.monotonic()-start)),flush=True)
        if a.smoke:
            for c in pipes:c.send(BASE)
            save('smoke.json',[r for batch in receive()for r in batch['rows']]);return
        for i in (a.pod_index,a.pod_index+10):
            name=FAMILIES[i][0];xs=[];ys=[];trials=[];rng=np.random.default_rng(250000+i)
            with open(out/(name+'-games.jsonl'),'w',buffering=1)as log:
                for iteration in range(32):
                    x,method=(initial(i),'seeded')if iteration==0 else bo.propose(xs,ys,rng,len(initial(i)))
                    cfg=configuration(i,x)
                    for c in pipes:c.send(cfg)
                    rows=[r for batch in receive()for r in batch['rows']];assert sorted(r['seed']for r in rows)==TRAIN
                    for r in rows:r.update(iteration=iteration+1,family=name);log.write(json.dumps(r)+'\n')
                    score=float(np.mean([r['gain']for r in rows]));xs.append(x.tolist());ys.append(score)
                    trials.append(dict(iteration=iteration+1,score=score,best=max(ys),config=cfg,x=x.tolist(),method=method))
                    save(name+'-trials.json',trials);print(json.dumps(dict(family=name,iteration=iteration+1,score=score,best=max(ys),elapsed=time.monotonic()-start)),flush=True)
            save(name+'-winner.json',trials[int(np.argmax(ys))])
        save('complete.json',dict(elapsed_seconds=time.monotonic()-start,games=12800));print('COMPLETE',flush=True)
    finally:
        for c in pipes:
            try:c.send(None)
            except (BrokenPipeError,EOFError):pass
        for p in actors:p.join(5)
        for p in actors:
            if p.is_alive():p.terminate();p.join()
if __name__=='__main__':main()
