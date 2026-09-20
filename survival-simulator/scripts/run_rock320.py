"""Durable central BO scheduler; all pods pull from one queue through SSH stdio.
Resume with the same --out. Completed jobs are never resubmitted. In-flight jobs
are recovered after a disconnected worker; failed games stop the campaign.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import asyncio,argparse,json,pathlib,time,sys,hashlib,fcntl
import numpy as np
from rock320_config import *
from tune_families10 import propose
REMOTE='/workspace/lucas-rock320'
KEY=str(pathlib.Path.home()/'.ssh/runpod_codex_team')
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
async def main(a):
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pods=json.loads((ROOT/'docs/rock-face-bo320/dispatch.json').read_text())
 if a.pods:pods=pods[:a.pods]
 manifest=dict(train=TRAIN,test=TEST,iterations=ITERATIONS,families=NAMES,space=SPACES,base=BASE,pilot=a.pilot,objective='mean full-game score',seed_policy=0,horizon=3000,normalization='inputs [0,1], GP targets centered and scaled each iteration',optimizer='Matern 5/2 GP, expected improvement, previous winner then 5 random startup trials',previous=PREVIOUS,test_caveat='Reused final maps; comparison benchmark, not a fresh holdout',pods=pods)
 if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
 else:save(out/'manifest.json',manifest)
 queue=asyncio.Queue();done=asyncio.Event();completed={};issued={};trials=[[]for _ in NAMES];start=time.monotonic();stage='pilot' if a.pilot else 'training'
 if (out/'jobs.jsonl').exists():
  for l in (out/'jobs.jsonl').read_text().splitlines():
   r=json.loads(l);completed[r['id']]=r
 log=(out/'jobs.jsonl').open('a',buffering=1)
 def submit(job):
  issued[job['id']]=job
  if job['id'] not in completed:queue.put_nowait(job)
 def job(stage,name,seed,cfg):return dict(id=f'{stage}/{name}/{seed}',seed=seed,configs=cfg)
 def trial(i,it):
  path=out/f'{NAMES[i]}-{it:02d}-config.json'
  if path.exists():data=json.loads(path.read_text())
  else:
   prior=trials[i];rng=np.random.default_rng(32000+i*1000+it)
   x,method=(initial(i),'previous training winner')if it==1 else propose([t['x']for t in prior],[t['score']for t in prior],rng,len(SPACES[i]))
   data=dict(x=list(x),method=method,config=PREVIOUS.copy() if it==1 else config(i,x),iteration=it);save(path,data)
  for seed in TRAIN:submit(job('train',f'{NAMES[i]}/{it}',seed,{NAMES[i]:data['config']}))
 def update():
  nonlocal stage
  if a.pilot:
   if len(completed)==len(issued):done.set()
   return
  for i,name in enumerate(NAMES):
   while len(trials[i])<ITERATIONS:
    it=len(trials[i])+1
    ids=[f'train/{name}/{it}/{seed}' for seed in TRAIN]
    if not all(k in completed for k in ids):break
    data=json.loads((out/f'{name}-{it:02d}-config.json').read_text());scores=[completed[k]['rows'][0]['score']for k in ids]
    data.update(score=float(np.mean(scores)),best=max([t['score']for t in trials[i]]+[float(np.mean(scores))]))
    trials[i].append(data);save(out/f'{name}-trials.json',trials[i])
    print(json.dumps(dict(family=name,iteration=it,mean=data['score'],best=data['best'])),flush=True)
    if it<ITERATIONS:trial(i,it+1)
  if all(len(t)==ITERATIONS for t in trials) and stage=='training':
   winners={name:max(t,key=lambda x:x['score'])for name,t in zip(NAMES,trials)}
   save(out/'frozen-winners.json',winners) # persisted BEFORE any held-out game
   stage='final'
   configs={name:w['config']for name,w in winners.items()};configs['expanded_food_baseline']=BASE;configs['previous_rock_face']=PREVIOUS
   # All three models on a given map execute on one worker/CPU. Rotate order.
   for seed in TEST:
    names=list(configs);shift=seed%len(names);names=names[shift:]+names[:shift]
    submit(job('final','paired',seed,{name:configs[name]for name in names}))
  if stage=='final' and all(f'final/paired/{s}'in completed for s in TEST):done.set()
  save(out/'progress.json',dict(stage=stage,trials={n:len(t)for n,t in zip(NAMES,trials)},completed_jobs=len(completed),completed_games=len(completed)+2*sum(k.startswith('final/') for k in completed),queued=queue.qsize(),elapsed_session=time.monotonic()-start,updated=time.time()))
 if a.pilot:
  for i,name in enumerate(['expanded_food_baseline']+NAMES):
   cfg=BASE if i==0 else config(i-1,initial(i-1))
   for seed in PILOT:submit(job('pilot',name,seed,{name:cfg}))
 else:
  for i in range(len(NAMES)):trial(i,1)
 update()
 async def connection(p):
  while not done.is_set():
   inflight={};proc=None
   try:
    cmd=f'cd {REMOTE}/survival-simulator && /workspace/lucas-families10/.venv/bin/python -u scripts/rock320_worker.py'
    proc=await asyncio.create_subprocess_exec('ssh','-i',KEY,'-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=10','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-p',str(p['port']),f"root@{p['host']}",cmd,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    hello=json.loads(await asyncio.wait_for(proc.stdout.readline(),60))
    if not hello.get('hello'):raise RuntimeError(hello)
    for rel,digest in hello['sources'].items():
     if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Source mismatch {rel}')
    build=out/f'pod-{p["index"]}-build.json'
    if build.exists() and not build.with_suffix('.before-sixth.json').exists():save(build.with_suffix('.before-sixth.json'),json.loads(build.read_text()))
    save(build,hello)
    sem=asyncio.Semaphore(32)
    async def send():
     while True:
      await sem.acquire();j=await queue.get()
      if j['id']in completed:sem.release();continue
      inflight[j['id']]=j;proc.stdin.write((json.dumps(j)+'\n').encode());await proc.stdin.drain()
    sender=asyncio.create_task(send())
    try:
     while not done.is_set():
      line=await asyncio.wait_for(proc.stdout.readline(),600)
      if not line:raise ConnectionError('Worker disconnected: '+(await proc.stderr.read()).decode()[-1000:])
      r=json.loads(line)
      if 'error'in r:raise RuntimeError(f"Game {r['id']}: {r['error']}")
      j=inflight.pop(r['id']);sem.release()
      assert set(x['model']for x in r['rows'])==set(j['configs']) and all(x['seed']==j['seed']for x in r['rows'])
      if r['id']not in completed:
       r['pod']=p['index'];log.write(json.dumps(r)+'\n');completed[r['id']]=r;update()
    finally:sender.cancel();await asyncio.gather(sender,return_exceptions=True)
   except RuntimeError as e:
    save(out/'ERROR.json',dict(error=str(e),pod=p['index'],time=time.time()));done.set();raise
   except Exception as e:
    print(f'Pod {p["index"]} reconnecting: {e}',flush=True)
    for j in inflight.values():
     if j['id']not in completed:queue.put_nowait(j)
    await asyncio.sleep(10)
   finally:
    if proc and proc.returncode is None:
     proc.stdin.close()
     try:await asyncio.wait_for(proc.wait(),10)
     except asyncio.TimeoutError:proc.terminate()
 tasks=[asyncio.create_task(connection(p))for p in pods]
 try:await asyncio.wait_for(done.wait(),28800)
 except asyncio.TimeoutError:
  save(out/'ERROR.json',dict(error='Eight-hour coordinator fail-safe reached; resume explicitly after inspection.'));done.set()
 for t in tasks:t.cancel()
 result=await asyncio.gather(*tasks,return_exceptions=True)
 log.close()
 if (out/'ERROR.json').exists():raise RuntimeError((out/'ERROR.json').read_text())
 save(out/'complete.json',dict(jobs=len(completed),games=sum(len(r['rows'])for r in completed.values()),elapsed_session=time.monotonic()-start))
 print('COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--pods',type=int,default=0);asyncio.run(main(ap.parse_args()))
