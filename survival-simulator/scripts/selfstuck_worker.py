"""SSH stdio worker. A bounded native-process pool consumes the shared queue."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,pathlib,hashlib,time,threading,platform
from multiprocessing import Pool
from evaluate_frozen1000 import one
import importlib.util
GENTLE_CORE=None
ROOT=pathlib.Path(__file__).resolve().parents[1]
LOCK=threading.Lock()
def emit(x):
 with LOCK:print(json.dumps(x),flush=True)
def gentle_one(name,seed,cfg):
 global GENTLE_CORE
 if GENTLE_CORE is None:
  path=ROOT/'vendor/gentle-migration/fastsim'
  spec=importlib.util.spec_from_file_location('gentlefastsim',path/'__init__.py',submodule_search_locations=[str(path)])
  module=importlib.util.module_from_spec(spec);sys.modules['gentlefastsim']=module;spec.loader.exec_module(module)
  from gentlefastsim.fastpolicy import PolicySimulationCore
  GENTLE_CORE=PolicySimulationCore
 sim=GENTLE_CORE(seed=seed,predators=True);sim.policy_init(0,cfg);wall=time.perf_counter_ns();cpu=time.process_time_ns()
 steps,peak,interface,policy,engine=sim._engine.run_policy(3000.,3000.,True)
 cpu=time.process_time_ns()-cpu;wall=time.perf_counter_ns()-wall;events=sim.pop_events()
 return dict(model=name,seed=seed,score=sim.env.score,survival=sim.env.time,steps=steps,peak=peak,ns_interface=interface,ns_policy=policy,ns_engine=engine,ns_loop_wall=wall,ns_loop_cpu=cpu,predation_deaths=sum(e[0]=='predator'for e in events),energy_deaths=sum(e[0]=='starvation'for e in events))
def run(job):
 start=time.monotonic();rows=[]
 for name,cfg in job['configs'].items():
  r=gentle_one(name,job['seed'],cfg)if name=='gentle_migration'else one((name,job['seed'],cfg));rows.append(r)
 return dict(id=job['id'],rows=rows,seconds=time.monotonic()-start)
def main():
 files=list((ROOT/'fastsim').glob('*.hpp'))+list((ROOT/'fastsim').glob('*.cpp'))+[ROOT/'models/self_stuck.hpp',ROOT/'scripts/selfstuck_config.py',ROOT/'scripts/selfstuck_worker.py',ROOT/'scripts/evaluate_frozen1000.py']
 files+=list((ROOT/'vendor/gentle-migration/fastsim').glob('*.hpp'))+list((ROOT/'vendor/gentle-migration/fastsim').glob('*.cpp'))
 emit(dict(hello=True,python=sys.version,platform=platform.platform(),sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},build=json.loads((ROOT/'fastsim/build-info-policy.json').read_text())))
 with Pool(32) as pool:
  for line in sys.stdin:
   job=json.loads(line)
   pool.apply_async(run,(job,),callback=emit,error_callback=lambda e,j=job:emit(dict(id=j['id'],error=repr(e))))
  pool.close();pool.join()
if __name__=='__main__':main()
