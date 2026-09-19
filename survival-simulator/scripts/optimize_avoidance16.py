"""16 avoidance families, independent Gaussian-process Bayesian optimization.
Native C++ engine AND policy; Python only schedules games and fits the GP.
Frozen split: train 1001..1008, validation 2001..2032, confirmation 3001..3064.
"""
import argparse, hashlib, json, math, os, pathlib, platform, sys, time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from multiprocessing import Pool
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nightsim.run import one

BASE=json.loads((ROOT/'docs/avoidance/native-baselines.json').read_text())['with_predators_best']
COMMON={'pred_r':(45,110),'pred_face_r':(70,180),'pred_sprint_r':(25,110)}
# Each family changes a mechanism, rather than just supplying 16 random settings.
FAMILIES=[
 ('sidestep','Tuned sideways escape',{}, {'pred_dodge_ang':(.6,1.8)}),
 ('radial','Walk or sprint directly away',{'pred_dodge_r':0},{}),
 ('look_away','Escape without turning gaze toward predator',{'pred_face':0},{'pred_dodge_ang':(.6,1.8)}),
 ('committed','Keep an escape heading for several ticks',{}, {'pred_dodge_hold':(1,5),'pred_dodge_ang':(.8,1.6)}),
 ('wall_clear','Deflect escapes around observed walls',{'pred_wallclear':1},{'pred_dodge_ang':(.8,1.6)}),
 ('shared','React to current group sightings',{'pred_share':1},{'pred_dodge_ang':(.8,1.6)}),
 ('closest_only','Forage when another visible agent is closer',{}, {'evade_closest':(25,65),'pred_dodge_ang':(.8,1.6)}),
 ('memory','Avoid orchard posts near remembered predators',{}, {'pred_avoid_w':(.03,.8),'pred_avoid_r':(80,250),'pred_avoid_t':(5,45)}),
 ('safe_births','Delay reproduction near a seen predator',{}, {'spawn_pred_r':(55,160),'pred_dodge_ang':(.8,1.6)}),
 ('sprint_reserve','Preserve energy needed for future sprinting',{}, {'sprint_floor':(5,100),'pred_dodge_ang':(.8,1.6)}),
 ('nearest_corner','Steer toward nearest corner, target ±10 degrees',{'corner_mode':1,'corner_tolerance':10},{'corner_budget':(.5,3),'corner_start_angle':(10,100),'corner_gaze':(.1,.5)}),
 ('sparse_corner','Steer toward one observed sparse corner, target ±10 degrees',{'corner_mode':2,'corner_tolerance':10},{'corner_budget':(.5,3),'corner_start_angle':(10,100),'corner_gaze':(.1,.5)}),
 ('cone_trigger','Tune facing-angle threshold for reacting',{}, {'pred_cone':(.2,.8),'pred_dodge_ang':(.8,1.6)}),
 ('three_step','Choose movement using three-step pursuit approximation',{'evade_search':1},{'evade_energy':(.05,3)}),
 ('energy_escape','Choose one-step cone exits with an energy price',{'evade_search':2},{'evade_energy':(.5,15)}),
 ('fruit_escape','Choose one-step cone exits favoring the orchard direction',{'evade_search':2},{'evade_energy':(.1,5),'evade_goal':(2,35)}),
]

def config(index,x):
    _,_,fixed,extra=FAMILIES[index]; ranges={**COMMON,**extra};kw={**BASE,**fixed}
    for (key,(lo,hi)),v in zip(ranges.items(),x):kw[key]=float(lo+(hi-lo)*v)
    if 'pred_dodge_hold' in kw:kw['pred_dodge_hold']=int(round(kw['pred_dodge_hold']))
    if fixed.get('pred_dodge_r')!=0:kw['pred_dodge_r']=max(kw['pred_r'],kw['pred_face_r'])
    return kw

def initial(index):
    ranges={**COMMON,**FAMILIES[index][3]};d={**BASE,'pred_dodge_hold':2,'evade_closest':45,'pred_avoid_w':.2,'pred_avoid_r':150,'pred_avoid_t':15,'spawn_pred_r':80,'sprint_floor':30,'corner_budget':2,'corner_start_angle':30,'corner_gaze':.25,'pred_cone':.5,'evade_energy':1,'evade_goal':10}
    return np.array([np.clip((d[k]-lo)/(hi-lo),0,1) for k,(lo,hi) in ranges.items()])

def propose(xs,ys,rng,dim):
    if len(xs)<6:return rng.random(dim),'random_startup'
    # Fixed Matern-5/2 covariance, standardized objective, Gaussian observation noise.
    # Expected improvement chooses among 2048 candidates; half explore globally,
    # half search near the incumbent. All preceding completed scores train the GP.
    x=np.array(xs);y=np.array(ys);y=(y-y.mean())/max(y.std(),1.)
    def kernel(a,b):
        r=np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(axis=2))/.4
        return (1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)
    K=kernel(x,x)+.15**2*np.eye(len(x));L=np.linalg.cholesky(K)
    c=np.vstack([rng.random((1024,dim)),np.clip(x[np.argmax(y)]+rng.normal(0,.15,(1024,dim)),0,1)])
    k=kernel(x,c);alpha=np.linalg.solve(L.T,np.linalg.solve(L,y));mu=k.T@alpha
    v=np.linalg.solve(L,k);sigma=np.sqrt(np.maximum(1-(v*v).sum(axis=0),1e-9))
    gain=mu-y.max()-.01;z=gain/sigma
    cdf=np.array([.5*(1+math.erf(a/math.sqrt(2))) for a in z])
    ei=gain*cdf+sigma*np.exp(-z*z/2)/math.sqrt(2*math.pi)
    return c[np.argmax(ei)],'gp_expected_improvement'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=32);ap.add_argument('--rounds',type=int,default=16);ap.add_argument('--deadline-seconds',type=int,default=12000);a=ap.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'games.jsonl').exists():raise SystemExit('Refusing to mix runs: use a fresh output directory')
    import nightsim
    sources=[ROOT/'nightsim/_nengine.cpp',ROOT/'nightsim/_npolicy.hpp',ROOT/'models/avoidance/native_corner.hpp',ROOT/'models/avoidance/native_escape_search.hpp',pathlib.Path(__file__)]
    binary=pathlib.Path(nightsim._engine.__file__)
    if binary.stat().st_mtime<max(p.stat().st_mtime for p in sources[:4]):raise SystemExit('Rebuild native binary')
    manifest={'baseline':BASE,'families':[{'id':x[0],'idea':x[1],'fixed':x[2],'ranges':{**COMMON,**x[3]}} for x in FAMILIES], 'train_seeds':list(range(1001,1009)),'validation_seeds':list(range(2001,2033)),'confirmation_seeds':list(range(3001,3065)),'rounds':a.rounds,'workers':a.workers,'horizon':3000,'predators':True,'optimizer':'independent Matern-5/2 GP expected improvement, 6 startup trials (including center), noise=0.15, lengthscale=0.4','python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    t0=time.time();rngs=[np.random.default_rng(90700+i) for i in range(16)];xs=[[]for _ in range(16)];ys=[[]for _ in range(16)];trials=[];games=[]
    def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
    with Pool(a.workers) as pool,open(out/'games.jsonl','a',buffering=1) as f:
        def evaluate(cfgs,seeds,stage):
            jobs=[(label,kw,s,3000.,True,250.) for s in seeds for label,kw in cfgs.items()]
            rows=[]
            for r in pool.imap_unordered(one,jobs):
                r['stage']=stage;rows.append(r);games.append(r);f.write(json.dumps(r)+'\n')
                if time.time()-t0>a.deadline_seconds:raise TimeoutError('Compute deadline reached; partial rows retained')
            return rows
        for r in range(a.rounds):
            cfgs={};pending=[]
            for i,fam in enumerate(FAMILIES):
                x,method=(initial(i),'center_startup') if r==0 else propose(xs[i],ys[i],rngs[i],len(initial(i)))
                label=f'{fam[0]}__{r:02d}';cfgs[label]=config(i,x);pending.append((i,label,x,method))
            rows=evaluate(cfgs,manifest['train_seeds'],'tune')
            for i,label,x,method in pending:
                score=float(np.mean([v['score'] for v in rows if v['label']==label]));xs[i].append(x.tolist());ys[i].append(score)
                trials.append({'family':FAMILIES[i][0],'round':r,'method':method,'score':score,'config':cfgs[label],'x':x.tolist()})
            save('trials.json',trials);print(json.dumps({'round':r+1,'games':len(games),'elapsed':round(time.time()-t0),'top_training':max(ys[i][-1] for i in range(16))}),flush=True)
        winners={fam[0]:config(i,xs[i][int(np.argmax(ys[i]))]) for i,fam in enumerate(FAMILIES)}
        save('tuned-configs.json',winners)
        valid=evaluate({'baseline':BASE,**winners},manifest['validation_seeds'],'validation')
        means={lab:float(np.mean([r['score']for r in valid if r['label']==lab])) for lab in ['baseline',*winners]}
        ranked=sorted(winners,key=lambda lab:means[lab],reverse=True);save('validation-ranking.json',means)
        save('confirmation-configs.json',{'baseline':BASE,**{lab:winners[lab] for lab in ranked[:2]}})
        print(json.dumps({'validation':means,'finalists':ranked[:2],'elapsed':round(time.time()-t0)}),flush=True)
        evaluate({'baseline':BASE,**{lab:winners[lab]for lab in ranked[:2]}},manifest['confirmation_seeds'],'confirmation')
    save('completion.json',{'games':len(games),'elapsed_seconds':time.time()-t0,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    print('COMPLETE',len(games),flush=True)
if __name__=='__main__':main()
