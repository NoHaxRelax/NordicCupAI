"""Bounded 64-game speed sample; drain only this task's BO broker, then resume."""
import os,signal,pathlib,json,time,subprocess,sys,random,argparse
from multiprocessing import Pool
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_frozen1000 import one
OUT=None
def broker():
 found={}
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   parts=(p/'cmdline').read_bytes().split(b'\0')
   if b'scripts/selfstuck_worker.py' in parts and(p/'cwd').resolve()==pathlib.Path('/workspace/lucas-selfstuck5/survival-simulator'):
    st=(p/'stat').read_text().split();found[int(p.name)]=int(st[3])
  except OSError:pass
 return [p for p,parent in found.items()if parent not in found],found
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='runpod32');ap.add_argument('--maps',type=int,default=32);ap.add_argument('--source',default='f31f586');args=ap.parse_args()
 OUT=ROOT/'docs/sharedfood-speed'/args.tag;OUT.mkdir(parents=True,exist_ok=True)
 assert not(OUT/'manifest.json').exists(),'Use a fresh sample tag'
 roots,found=broker();assert len(roots)==1,(roots,found);pid=roots[0]
 (OUT/'broker.json').write_text(json.dumps({'pid':pid,'start':time.time(),'note':'Only parent broker suspended; running game children drain normally. No simulation CPU timers suspended.'}))
 # Independent deadman ensures broker resumes if this SSH session is lost.
 subprocess.Popen([sys.executable,'-c',f'import os,signal,time;time.sleep(540);os.kill({pid},signal.SIGCONT)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  os.kill(pid,signal.SIGSTOP)
  stable=0;deadline=time.monotonic()+120
  while stable<3:
   active=False
   for child in found:
    if child==pid:continue
    try:active|=pathlib.Path(f'/proc/{child}/stat').read_text().split()[2]=='R'
    except OSError:pass
   stable=0 if active else stable+1
   if time.monotonic()>deadline:raise TimeoutError('BO workers did not drain within 120s')
   time.sleep(1)
  cfg=json.loads((ROOT/'docs/sharedfood20/final/shard-0/manifest.json').read_text())['configs']
  names=['shared_food_control','congestion_pricing'];seeds=list(range(30001,30001+args.maps));jobs=[(n,s,cfg[n])for s in seeds for n in names];random.Random(30000).shuffle(jobs)
  (OUT/'manifest.json').write_text(json.dumps({'source_commit':args.source,'seeds':seeds,'configs':{n:cfg[n]for n in names},'workers':32,'profile':True,'build':json.loads((ROOT/'fastsim/build-info-policy.json').read_text())},indent=2))
  start=time.monotonic()
  with Pool(32)as pool,(OUT/'games.jsonl').open('w',buffering=1)as f:
   it=pool.imap_unordered(one,jobs)
   for _ in jobs:
    row=it.next(timeout=max(1,360-(time.monotonic()-start)));f.write(json.dumps(row)+'\n')
  (OUT/'complete.json').write_text(json.dumps({'games':len(jobs),'elapsed':time.monotonic()-start}))
 finally:
  os.kill(pid,signal.SIGCONT)
  (OUT/'broker-resumed.json').write_text(json.dumps({'pid':pid,'time':time.time()}))
