"""Require exact action traces before reporting speed changes."""
import pathlib,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
out=ROOT/'docs/sharedfood-speed';checks=[]
for seed in [19001,19099,19038]:
 for model in ['shared_food_control','congestion_pricing']:
  before,after=[json.loads((out/f'trace-{v}-{model}-{seed}.json').read_text())for v in ['before','after']]
  for k in ['action_digest','score','time','ticks']:assert before[k]==after[k],(seed,model,k,before[k],after[k])
  checks.append(dict(seed=seed,model=model,ticks=before['ticks'],action_digest=before['action_digest'],score=before['score'],exact=True))
(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
lines=['# Sharing and congestion pricing: behavior-preserving optimization','',
 'Branch `codex/sharedfood-speed`, based on frozen `fba2549`. No policy parameters or game rules changed. Runpod BO uses its own frozen code; these changes were not deployed there.','',
 '## Measured full-game native runtime','',
 'Same laptop, seed 19001, same frozen configurations, normal energy/predators, horizon 3000 or extinction. Two benchmark processes ran concurrently; these are single-run measurements, not hardware-independent latency estimates. Initialization excluded; phase profiling enabled in both builds.','',
 '| Policy | Before | After | Speedup | Before CPU µs/tick | After CPU µs/tick |','|---|---:|---:|---:|---:|---:|']
for model in ['shared_food_control','congestion_pricing']:
 a,b=[json.loads((out/f'laptop-{v}-{model}.json').read_text())for v in ['before','after']]
 assert all(a[k]==b[k]for k in ['score','time','ticks'])
 lines.append(f'| {model} | {a["wall_seconds"]:.2f}s | {b["wall_seconds"]:.2f}s | {a["wall_seconds"]/b["wall_seconds"]:.2f}× | {a["cpu_seconds"]*1e6/a["ticks"]:.1f} | {b["cpu_seconds"]*1e6/b["ticks"]:.1f} |')
lines+=['','## What changed','',
 '- Wall-based relocalization now looks up nearby wall-direction buckets rather than scanning every wall for each observed edge. Candidate indices are sorted back into their original order, preserving tie breaks and summation order. New/deleted/transformed walls invalidate the index; timestamp updates do not.',
 '- Direction matching uses the existing squared-distance rejection with precise distance fallback near the threshold.',
 '- Shared-wall sorting computes each distance once instead of recalculating it for every comparator invocation. Stable order and the same 150-wall cap remain.',
 '- Congestion counts are cached while tree assignments are unchanged. Reassigning an agent clears the cache; each querying agent is excluded exactly as before.',
 '- A repeated nearest-candidate distance calculation is reused.','',
 '## Verification','',
 f'All actions matched exactly on every tick in **six full policy/map runs ({sum(c["ticks"]for c in checks):,} ticks)**: both policies on seeds 19001, 19099 and 19038. Scores, lifetimes and tick counts also match exactly. The digest includes agent IDs, movement distances/directions, turn angles and reproduction decisions. The observation-only compile boundary check passed.',
 '',
 'This is strong regression evidence, not an exhaustive proof across every map. Speedup has not been remeasured on the 1,000-map Runpod panel. The original functionality remains experimental: these optimizations do not fix the previously measured full-game score deficit.',
 '',
 'Profiling initially attributed 14.58 of 18.30 wall seconds to observation processing on the first 500 simulated seconds of sharing-only seed 19001. After optimization, observation time was 1.76 seconds and total wall time 4.21 seconds. The wall-direction lookup addresses the dominant repeated work.',
 '',
 'PC checks were stopped when access was revoked; all reported results and completed verification here are from the laptop. Raw JSON files and benchmark_sharedfood_speed.py allow reproduction.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Verified',len(checks),'full-game action traces')
