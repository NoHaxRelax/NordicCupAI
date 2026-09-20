# Experimental seed-aware survival policy

Target: average score at least 2200 on 128 fresh paired seeds. **Not achieved.** This is policy research only; no seed recovery or competition API client.

Use Python 3.12 and NumPy 2.3.5 on Linux. NumPy compiled math kernels are bound once; simulation and policy run in C++. Pin PYTHONPATH to that environment for the embedded interpreter. Compile with `python build.py`, then run `bench/policy_bench configs/orchard.json SEED MODE 3000 OUTPUT 180`. Model availability is delayed 180 simulated seconds in reported comparisons. Current-state engine truth is an upper bound, not an immutable future schedule or live validation claim.

Baseline mode8 remains selected. On wave19, 128 fresh paired seeds, it averaged1764.30 versus1749.75 for exact-predator evasion27. The earlier32-seed gain of193.95 did not replicate; mode27 is not promoted. Pre-activation hashes matched every pair.

Experimental modes35/36/37 try one-tick predator responses to escape candidates, preserving Lucas actions unless predicted clearance falls below2/12/25px. All failed the complete16-seed wave17 batch: deltas-79.73/-107.35/-131.93. They omit births, meals and cross-predator kills within lookahead and are not promoted. Dryrun38 exercised386487candidate predictions and preserved the full baseline score, duration, food, deaths and prefix hash exactly.

Numerical portability resolved: a full seed1283794550 baseline run on mypc matches Runpod under NumPy2.3.5 exactly. Earlier mypc NumPy1.26.4 evidence is retained as historical exploratory data and must not be pooled with canonical results. Full completed replays are retained locally; GitHub includes compact frozen sources and paired result evidence.
