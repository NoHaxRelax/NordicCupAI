"""Summarize the full-game runtime sample; incomplete samples are rejected."""
import pathlib,json,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];out=ROOT/'docs/sharedfood-speed'/(sys.argv[1]if len(sys.argv)>1 else 'runpod32')
rows=[json.loads(l)for l in(out/'games.jsonl').read_text().splitlines()]
assert json.loads((out/'complete.json').read_text())['games']==len(rows)
assert len(rows)==2*len(json.loads((out/'manifest.json').read_text())['seeds'])
manifest=json.loads((out/'manifest.json').read_text());result={}
for model in manifest['configs']:
 rr=sorted([r for r in rows if r['model']==model],key=lambda r:r['seed']);assert [r['seed']for r in rr]==manifest['seeds']
 wall=np.array([r['ns_loop_wall']/1e9 for r in rr]);cpu=np.array([r['ns_loop_cpu']/1e9 for r in rr]);steps=sum(r['steps']for r in rr)
 boot=np.random.default_rng(30000).choice(wall,(10000,len(wall))).mean(1)
 result[model]=dict(n=len(rr),mean_seconds=float(wall.mean()),mean_ci95=np.quantile(boot,[.025,.975]).tolist(),median_seconds=float(np.median(wall)),p90_seconds=float(np.quantile(wall,.9)),mean_cpu_seconds=float(cpu.mean()),cpu_us_per_tick=float(cpu.sum()*1e6/steps),mean_survival=float(np.mean([r['survival']for r in rr])))
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
lines=['# Runpod runtime sample: committed sharing optimizations','',
 f'Source `{manifest["source_commit"]}`; {len(manifest["seeds"])} fresh seeds {min(manifest["seeds"])}–{max(manifest["seeds"])} per policy, ordinary full games with predators/energy, 3000-second horizon or extinction. One existing 32-vCPU Runpod CPU pod, 32 game workers, randomized job order. BO broker drained and resumed; BO game processes were not suspended.','',
 '| Policy | Mean seconds/game | Bootstrap 95% CI | Median | P90 | CPU µs/tick |','|---|---:|---|---:|---:|---:|']
for name,r in result.items():
 lo,hi=r['mean_ci95'];lines.append(f'| {name} | {r["mean_seconds"]:.2f} | {lo:.2f}–{hi:.2f} | {r["median_seconds"]:.2f} | {r["p90_seconds"]:.2f} | {r["cpu_us_per_tick"]:.1f} |')
lines+=['','Times cover the native game loop, excluding map initialization; profiling is enabled. Intervals describe map-sampling uncertainty on this host/load, not between-host or repeated-measurement uncertainty. This sample is not a new 1,000-map runtime average. Later jobs may experience lower concurrency as the sample drains.','', 'Only committed behavior-preserving optimizations were used; no new tuning or reduced game horizon. Source build provenance, raw rows and BO-broker resumption record are stored here.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
