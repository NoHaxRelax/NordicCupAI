"""Copy frozen sources into a new directory on existing pods, then build locally."""
import pathlib,json,tarfile,subprocess,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/late-gated5';ARCHIVE=OUT/'source.tar.gz'
with tarfile.open(ARCHIVE,'w:gz') as t:
 for folder in ('fastsim','scripts','models'):
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ('.py','.cpp','.hpp','.inc'):t.add(p,arcname=str(p.relative_to(ROOT)))
 for rel in ('docs/localfood20/pod-7/expanded_local_food-winner.json','docs/selfstuck5/run/frozen-winners.json','docs/late-gated5/source-presets.json'):
  t.add(ROOT/rel,arcname=rel)

def deploy(p):
 cmd='mkdir -p /workspace/lucas-late-gated5/survival-simulator && cd /workspace/lucas-late-gated5/survival-simulator && tar xzf - && /workspace/lucas-families10/.venv/bin/python fastsim/build.py > ../build.log 2>&1 && /workspace/lucas-families10/.venv/bin/python fastsim/build_policy.py >> ../build.log 2>&1 && /workspace/lucas-families10/.venv/bin/python fastsim/check_boundary.py >> ../build.log 2>&1'
 with ARCHIVE.open('rb') as f:
  r=subprocess.run(['ssh','-i','/home/Ucals/.ssh/runpod_codex_team','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=10','-p',str(p['port']),'root@'+p['host'],cmd],stdin=f,capture_output=True,text=False,timeout=240)
 result=dict(pod=p['index'],returncode=r.returncode,error=r.stderr.decode()[-1000:]);print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:results=list(pool.map(deploy,json.loads((OUT/'dispatch.json').read_text())))
(OUT/'deployment.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(x['returncode']==0 for x in results)
