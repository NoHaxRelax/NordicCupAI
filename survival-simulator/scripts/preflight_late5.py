"""Run default and always-active full games on one pod at 32-worker concurrency."""
import pathlib,json,subprocess,shlex,statistics
R=pathlib.Path(__file__).resolve().parents[1];D=R/'docs/late-gated5';p=json.loads((D/'dispatch.json').read_text())[0]
script='''import sys,json,random
from multiprocessing import Pool
sys.path.insert(0,'/workspace/lucas-late-gated5/survival-simulator/scripts')
from late5_config import BASE,NAMES,PILOT,config,initial
from late5_worker import run
jobs=[]
for i,n in enumerate(NAMES):
 for mode in ['default','always']:
  c=config(i,initial(i))
  if mode=='always':c.update(gate_ticks=0,gate_logic=0,gate_persistence=1)
  for seed in PILOT:jobs.append(dict(id=n+'/'+mode+'/'+str(seed),seed=seed,configs={n:c},mode=mode))
for seed in PILOT:jobs.append(dict(id='baseline/default/'+str(seed),seed=seed,configs={'expanded_food_baseline':BASE},mode='default'))
random.Random(42000).shuffle(jobs)
with Pool(32) as pool:
 for result in pool.imap_unordered(run,jobs):print(json.dumps(result),flush=True)
'''
cmd='/workspace/lucas-families10/.venv/bin/python -c '+shlex.quote(script)
with (D/'preflight-games.jsonl').open('w')as f:subprocess.run(['ssh','-i','/home/Ucals/.ssh/runpod_codex_team','-o','BatchMode=yes','-o','UpdateHostKeys=no','-p',str(p['port']),'root@'+p['host'],cmd],stdout=f,check=True,timeout=900)
rows=[json.loads(l)for l in(D/'preflight-games.jsonl').read_text().splitlines()];assert len(rows)==352 and len({r['id']for r in rows})==352
summary={}
for j in rows:
 n,mode,seed=j['id'].split('/');summary.setdefault(n,{}).setdefault(mode,[]).append(j['rows'][0]['game_seconds'])
for n,modes in summary.items():
 for mode,values in modes.items():
  assert len(values)==32;modes[mode]=dict(mean=statistics.mean(values),p90=sorted(values)[28],max=max(values),eligible=statistics.mean(values)<20.)
summary={'models':summary,'passed':all(m['eligible']for v in summary.values()for m in v.values()),'pod':p,'workers':32,'games':352}
(D/'preflight-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if not summary['passed']:raise SystemExit('Pre-BO speed check failed; optimize or exclude before launching BO')
