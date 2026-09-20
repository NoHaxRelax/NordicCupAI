"""Paired local-only burst experiments. Never invokes a hosted evaluation API."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,pathlib,sys,time,random
from multiprocessing import Pool
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=ROOT/'oscar-overnight-cpp'
sys.path.insert(0,str(SRC/'nightsim/serve'))
from sim_harvest import one
BASE=json.loads((SRC/'nightsim/serve/pred_best.json').read_text())
HV=dict(enabled=True,budget=200000,max_harvests=100,sacrifice_mode='predict_contact',cooldown=50,contact_margin=1.0)
VARIANTS={
 'oscar200k':(BASE,HV),
 'burst400k':(BASE,dict(HV,budget=400000)),
 'fast400k':(BASE,dict(HV,budget=400000,cooldown=0)),
 'lowpop400k':(BASE,dict(HV,budget=400000,cooldown=0,min_free=2)),
 'fast1m':(BASE,dict(HV,budget=1000000,cooldown=0)),
 'biome400k':(dict(BASE,bio_w=8),dict(HV,budget=400000,cooldown=0)),
 'population400k':(dict(BASE,cap_max=50,cap_mult=.8,cap_min=4),dict(HV,budget=400000,cooldown=0)),
 'birth400k':(dict(BASE,breed_reserve=120,low_pop_reserve=120,heir_reserve=160),dict(HV,budget=400000,cooldown=0)),
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--start',type=int,default=91001);p.add_argument('--count',type=int,default=32);p.add_argument('--workers',type=int,default=12);p.add_argument('--names',default=','.join(VARIANTS));a=p.parse_args()
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);names=a.names.split(',')
 manifest=dict(start=a.start,count=a.count,variants={n:VARIANTS[n] for n in names},horizon=3000,score_ceiling=0,local_only=True)
 m=out/'manifest.json'
 if m.exists(): assert json.loads(m.read_text())==json.loads(json.dumps(manifest))
 else:m.write_text(json.dumps(manifest,indent=2))
 f=out/'games.jsonl';done=set()
 if f.exists():done={(r['label'],r['seed']) for r in map(json.loads,f.read_text().splitlines())}
 jobs=[(n,VARIANTS[n][0],s,3000.,VARIANTS[n][1],0.,0.) for s in range(a.start,a.start+a.count) for n in names if (n,s) not in done];random.Random(a.start).shuffle(jobs);t=time.monotonic()
 with Pool(a.workers) as pool,f.open('a',buffering=1) as dest:
  for i,r in enumerate(pool.imap_unordered(one,jobs),1):
   dest.write(json.dumps(r)+'\n')
   if i%10==0:print(f'{i}/{len(jobs)} elapsed={time.monotonic()-t:.1f}',flush=True)
 print('DONE',flush=True)
if __name__=='__main__':main()
