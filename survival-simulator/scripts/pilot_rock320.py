"""Same-pod native-loop before/after timing and outcome parity on 32 full games."""
import pathlib,json,subprocess,shlex,statistics
r=pathlib.Path(__file__).resolve().parents[1];out=r/'docs/rock-face-bo320';pod=json.loads((out/'dispatch.json').read_text())[0]
script='''import sys,json
from multiprocessing import Pool
root=sys.argv[1];sys.path.insert(0,root+'/scripts')
from evaluate_frozen1000 import one
cfg=json.load(open('/workspace/lucas-rock320/survival-simulator/docs/rock-face-bo320/check-config.json'))
with Pool(32) as p:
 for row in p.imap_unordered(one,[('rock_face_steer',s,cfg)for s in range(25001,25033)]):print(json.dumps(row),flush=True)
'''
for version,root in [('before','/workspace/lucas-selfstuck5/survival-simulator'),('after','/workspace/lucas-rock320/survival-simulator')]:
 cmd='/workspace/lucas-families10/.venv/bin/python -c '+shlex.quote(script)+' '+shlex.quote(root)
 with (out/f'pilot-{version}.jsonl').open('w')as f:
  subprocess.run(['ssh','-i','/home/Ucals/.ssh/runpod_codex_team','-o','BatchMode=yes','-o','UpdateHostKeys=no','-p',str(pod['port']),'root@'+pod['host'],cmd],stdout=f,check=True,timeout=240)
 print(version,'complete',flush=True)
a={x['seed']:x for x in map(json.loads,(out/'pilot-before.jsonl').read_text().splitlines())};b={x['seed']:x for x in map(json.loads,(out/'pilot-after.jsonl').read_text().splitlines())};assert set(a)==set(b)==set(range(25001,25033))
assert all(a[s][k]==b[s][k]for s in a for k in ('score','survival','steps','peak','predation_deaths','energy_deaths'))
summary={v:{'mean_loop_seconds':statistics.mean(x['ns_loop_wall']/1e9 for x in rows.values()),'mean_loop_cpu_seconds':statistics.mean(x['ns_loop_cpu']/1e9 for x in rows.values()),'loop_cpu_us_per_tick':sum(x['ns_loop_cpu']for x in rows.values())/sum(x['steps']for x in rows.values())/1000}for v,rows in [('before',a),('after',b)]};summary.update(exact_outcomes=32,pod=pod)
(out/'pilot-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
