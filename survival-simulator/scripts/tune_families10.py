"""One CPU pod per family: 30 GP trials, then 100 untouched paired maps.

Only scheduling/GP fitting uses Python; every policy tick and game uses C++.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, hashlib, json, math, pathlib, platform, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FULL = json.loads((ROOT/'models/best_policies/rank1_pod-03.json').read_text())
BASE = {k: float('inf') if v is None else v for k,v in FULL['orchard'].items()
        if k not in ('dump_food','dump_slack','heirs','hungry','old_dump','wait_tol')}
BASE.update(FULL['evasion'])
FAMILIES = [
 ('sidestep', 'Sideways escape', {}, {'pred_r':(55,150),'pred_sprint_r':(30,100),'pred_dodge_ang':(.6,1.8),'pred_face_r':(60,160)}),
 ('radial', 'Straight retreat', {'pred_dodge_r':0}, {'pred_r':(55,150),'pred_sprint_r':(30,110),'pred_face_r':(60,160)}),
 ('gaze', 'Watch predator while escaping', {'pred_gaze':1}, {'pred_r':(65,160),'pred_sprint_r':(30,110),'pred_dodge_ang':(.6,1.8),'pred_turn_max':(.2,1.5)}),
 ('cone', 'Ignore predators facing away outside hearing', {'pred_cone_gate':1}, {'pred_r':(65,180),'pred_sprint_r':(35,110),'pred_dodge_ang':(.6,1.8),'pred_cone_margin':(0,.4)}),
 ('wall', 'Break sight around observed wall corners', {'pred_wall_escape':1}, {'pred_r':(75,180),'pred_sprint_r':(35,110),'pred_wall_look':(10,55),'pred_wall_reward':(20,180)}),
 ('population', 'Match population to declining food', {}, {'cap_mult':(.12,.65),'cap_max':(10,45),'cap_min':(2,8),'tree_half':(250,1100)}),
 ('breeding', 'Energy reserves for reproduction', {}, {'breed_reserve':(120,330),'breed_reserve_late':(120,330),'low_pop_reserve':(105,240),'emergency_reserve':(80,160)}),
 ('heirs', 'Select and feed replacement generations', {}, {'heir_age':(35,85),'heir_reserve':(130,330),'heir_slack':(0,.3),'fit_energy':(0,1.5),'fit_speed':(0,1.5)}),
 ('harvest', 'Fruit ripeness and travel cost', {}, {'ripen_wait':(5,35),'fruit_reach':(80,340),'dist_pen':(.02,.4),'rot_margin':(35,49)}),
 ('exploration', 'Explore versus keep harvesting', {}, {'explore_energy':(100,300),'explore_radius':(150,700),'watch_patience':(5,80),'switch_gain':(20,180),'min_stay':(5,60)}),
]
TRAIN = list(range(6001,6033))
TEST = list(range(7001,7101))
ITERATIONS = 30
STARTUP = 6
RNG_SEED = 91000
DEADLINE = 1200
EXTRA_SOURCES = []
CONTROLS = {0: {'baseline': BASE}}

def initial(i):
    extra={'pred_wall_look':30,'pred_wall_reward':80,'pred_cone_margin':.1}
    return np.array([np.clip((extra.get(k,BASE.get(k,lo))-lo)/(hi-lo),0,1)
                     for k,(lo,hi) in FAMILIES[i][3].items()])

def config(i,x):
    _,_,fixed,ranges=FAMILIES[i]; c={**BASE,**fixed}
    c.update({k:float(lo+(hi-lo)*v) for (k,(lo,hi)),v in zip(ranges.items(),x)})
    if i < 5 and i != 1: c['pred_dodge_r']=c['pred_r']
    return c

def propose(xs,ys,rng,dim):
    if len(xs)<STARTUP:return rng.random(dim),'random_startup'
    x=np.array(xs); y=np.array(ys); y=(y-y.mean())/max(y.std(),1.)
    def kernel(a,b):
        r=np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(axis=2))/.4
        return (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
    L=np.linalg.cholesky(kernel(x,x)+.15**2*np.eye(len(x)))
    c=np.vstack([rng.random((1024,dim)),np.clip(x[np.argmax(y)]+rng.normal(0,.15,(1024,dim)),0,1)])
    k=kernel(x,c);mu=k.T@np.linalg.solve(L.T,np.linalg.solve(L,y))
    v=np.linalg.solve(L,k);sigma=np.sqrt(np.maximum(1-(v*v).sum(axis=0),1e-9))
    gain=mu-y.max()-.01;z=gain/sigma
    cdf=np.array([.5*(1+math.erf(a/math.sqrt(2)))for a in z])
    ei=gain*cdf+sigma*np.exp(-z*z/2)/math.sqrt(2*math.pi)
    return c[np.argmax(ei)],'gp_expected_improvement'

def one(job):
    from fastsim.fastpolicy import PolicySimulationCore
    seed,cfg,stage,iteration=job;t=time.monotonic()
    sim=PolicySimulationCore(seed=seed,predators=True);sim.policy_init(0,cfg)
    steps,peak,*_=sim.run_policy(3000.)
    return dict(seed=seed,stage=stage,iteration=iteration,score=sim.env.score,
                survival=sim.env.time,peak=peak,steps=steps,seconds=time.monotonic()-t)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=[f[0]for f in FAMILIES],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=32)
    ap.add_argument('--deadline',type=int,default=DEADLINE);ap.add_argument('--pilot',action='store_true')
    a=ap.parse_args();i=[f[0]for f in FAMILIES].index(a.family);out=pathlib.Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Use a fresh output directory')
    def save(name,data):
        tmp=out/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/name)
    sources=list((ROOT/'fastsim').glob('*.hpp'))+list((ROOT/'fastsim').glob('*.cpp'))+[pathlib.Path(__file__)]+EXTRA_SOURCES
    save('manifest.json',dict(family=FAMILIES[i],train_seeds=TRAIN,test_seeds=TEST,
         iterations=ITERATIONS,policy_seed=0,horizon=3000,predators=True,objective='mean score',
         base=BASE,python=sys.version,numpy=np.__version__,platform=platform.platform(),
         source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},
         build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text()),
         optimizer=f'Matern5/2 GP expected improvement; 1 center + {STARTUP-1} random + {ITERATIONS-STARTUP} EI',
         start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    xs=[];ys=[];trials=[];rng=np.random.default_rng(RNG_SEED+i);start=time.monotonic()
    with Pool(a.workers) as pool,open(out/'games.jsonl','w',buffering=1) as log:
        def evaluate(cfg,seeds,stage,it):
            results=pool.imap_unordered(one,[(s,cfg,stage,it)for s in seeds]);rows=[]
            for _ in seeds:
                remaining=a.deadline-(time.monotonic()-start)
                if remaining<=0:raise TimeoutError('Budget deadline; partial results saved')
                r=results.next(timeout=remaining);rows.append(r);log.write(json.dumps(r)+'\n')
            return rows
        if a.pilot:
            rows=evaluate(BASE,TRAIN,'pilot',0);save('pilot.json',dict(rows=rows,elapsed=time.monotonic()-start));return
        for it in range(ITERATIONS):
            x,method=(initial(i),'center')if it==0 else propose(xs,ys,rng,len(initial(i)))
            cfg=config(i,x);rows=evaluate(cfg,TRAIN,'train',it+1)
            score=float(np.mean([r['score']for r in rows]));xs.append(x.tolist());ys.append(score)
            trials.append(dict(iteration=it+1,score=score,best=max(ys),config=cfg,method=method,x=x.tolist()))
            save('trials.json',trials);print(json.dumps(dict(family=a.family,iteration=it+1,score=score,best=max(ys),elapsed=time.monotonic()-start)),flush=True)
        winner=trials[int(np.argmax(ys))];save('winner.json',winner)
        rows=evaluate(winner['config'],TEST,'test',winner['iteration'])
        scores=np.array([r['score']for r in sorted(rows,key=lambda r:r['seed'])])
        boot=np.random.default_rng(7100).choice(scores,(10000,len(TEST))).mean(axis=1)
        save('summary.json',dict(family=a.family,n=len(rows),mean=float(scores.mean()),
             median=float(np.median(scores)),std=float(scores.std(ddof=1)),ci95=np.quantile(boot,[.025,.975]).tolist(),
             training_start=ys[0],training_best=max(ys),best_iteration=winner['iteration'],
             elapsed_seconds=time.monotonic()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        for stage,control in CONTROLS.get(i,{}).items():
            save(stage+'-config.json',control)
            evaluate(control,TEST,stage,0)
        save('complete.json',dict(elapsed_seconds=time.monotonic()-start))
        print('COMPLETE',flush=True)
if __name__=='__main__':main()
