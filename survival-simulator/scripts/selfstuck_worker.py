"""SSH stdio worker. A bounded native-process pool consumes the shared queue."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,pathlib,hashlib,time,threading,platform
from multiprocessing import Pool
from evaluate_frozen1000 import one
ROOT=pathlib.Path(__file__).resolve().parents[1]
LOCK=threading.Lock()
def emit(x):
 with LOCK:print(json.dumps(x),flush=True)
def run(job):
 start=time.monotonic();rows=[]
 for name,cfg in job['configs'].items():
  r=one((name,job['seed'],cfg));rows.append(r)
 return dict(id=job['id'],rows=rows,seconds=time.monotonic()-start)
def main():
 files=list((ROOT/'fastsim').glob('*.hpp'))+list((ROOT/'fastsim').glob('*.cpp'))+[ROOT/'models/self_stuck.hpp',ROOT/'scripts/selfstuck_config.py',ROOT/'scripts/selfstuck_worker.py',ROOT/'scripts/evaluate_frozen1000.py']
 emit(dict(hello=True,python=sys.version,platform=platform.platform(),sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text())))
 with Pool(32) as pool:
  for line in sys.stdin:
   job=json.loads(line)
   pool.apply_async(run,(job,),callback=emit,error_callback=lambda e,j=job:emit(dict(id=j['id'],error=repr(e))))
  pool.close();pool.join()
if __name__=='__main__':main()
