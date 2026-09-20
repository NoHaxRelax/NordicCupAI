# Sharing and congestion pricing: behavior-preserving optimization

Branch `codex/sharedfood-speed`, based on frozen `fba2549`. No policy parameters or game rules changed. Runpod BO uses its own frozen code; these changes were not deployed there.

## Measured full-game native runtime

Same laptop, seed 19001, same frozen configurations, normal energy/predators, horizon 3000 or extinction. Two benchmark processes ran concurrently; these are single-run measurements, not hardware-independent latency estimates. Initialization excluded; phase profiling enabled in both builds.

| Policy | Before | After | Speedup | Before CPU µs/tick | After CPU µs/tick |
|---|---:|---:|---:|---:|---:|
| shared_food_control | 46.37s | 8.23s | 5.63× | 3838.3 | 700.5 |
| congestion_pricing | 34.05s | 8.32s | 4.09× | 2241.5 | 563.2 |

## What changed

- Wall-based relocalization now looks up nearby wall-direction buckets rather than scanning every wall for each observed edge. Candidate indices are sorted back into their original order, preserving tie breaks and summation order. New/deleted/transformed walls invalidate the index; timestamp updates do not.
- Direction matching uses the existing squared-distance rejection with precise distance fallback near the threshold.
- Shared-wall sorting computes each distance once instead of recalculating it for every comparator invocation. Stable order and the same 150-wall cap remain.
- Congestion counts are cached while tree assignments are unchanged. Reassigning an agent clears the cache; each querying agent is excluded exactly as before.
- A repeated nearest-candidate distance calculation is reused.

## Verification

All actions matched exactly on every tick in **six full policy/map runs (109,229 ticks)**: both policies on seeds 19001, 19099 and 19038. Scores, lifetimes and tick counts also match exactly. The digest includes agent IDs, movement distances/directions, turn angles and reproduction decisions. The observation-only compile boundary check passed.

This is strong regression evidence, not an exhaustive proof across every map. Speedup has not been remeasured on the 1,000-map Runpod panel. The original functionality remains experimental: these optimizations do not fix the previously measured full-game score deficit.

Profiling initially attributed 14.58 of 18.30 wall seconds to observation processing on the first 500 simulated seconds of sharing-only seed 19001. After optimization, observation time was 1.76 seconds and total wall time 4.21 seconds. The wall-direction lookup addresses the dominant repeated work.

PC checks were stopped when access was revoked; all reported results and completed verification here are from the laptop. Raw JSON files and benchmark_sharedfood_speed.py allow reproduction.
