# Seed-aware C++ survival policy checkpoint

The current best policy is mode144: Lucas orchard behavior with the synchronized one-step predictive safety intervention and child-priority 60. The model becomes available after a configurable **180 simulated seconds**; before activation the controller uses only the public observation stream. The C++ harness checks that pre-activation action hashes match across paired modes and that birth forecasts do not mutate the live RNG.

The fresh 500-seed confirmation scored **2076.2007 mean** against baseline mode46 at 1889.0477, for a paired gain of 187.1530. It had 363 wins and 137 losses, bootstrap 95% interval [155.6052, 218.5773], zero failures, zero cancellations, zero pre-activation hash conflicts, and 182833 verified birth forecasts. The 2200 target is not yet achieved.

Wave68 tested a turn-preserving safety scoring variant on 16 fresh pairs and lost by 186.2055, so it is not promoted. All experiments retain separate manifests, source hashes, and replay archives locally; compact summaries are included in `evidence/`.

Build with Python 3.12 and the pinned native environment, then run `bench/policy_bench` with the manifest's delay parameter. No evaluation or validation API client is included.

Wave69 tested exact-resource fruit allocation variants on 16 fresh pairs; all three lost to mode144 by 122–180 points and were rejected.

Wave70 tested fruit-claim conditional safety triggers on 16 fresh pairs; all three lost to mode144 by 73–159 points and were rejected.

Wave72 tested current-heading approaching-predator safety conditions on 16 fresh pairs; both variants lost to mode144 by 236–258 points and were rejected.

Wave73 tested stronger safety displacement and energy penalties on 16 fresh pairs; all four variants lost to mode144 by 49–203 points and were rejected.

Wave77 tested a cloned-engine future-fruit schedule on 8 fresh pairs; both horizons were harmful and rejected.

Wave78 tested denser and local dodge headings on 16 fresh pairs; both were strongly harmful and rejected.

Wave79 tested half-speed safety escape candidates on 16 fresh pairs; both were harmful and rejected.

Wave80 tested smaller Lucas predator-dodge radii and angles on 16 fresh pairs; all four were harmful and rejected.

Wave81: complete 16-pair predator-dodge ablation; all variants rejected versus mode144; 80 replays retained locally, not included in GitHub checkpoint.

Wave83: corrected fruit-readiness timing screen; all variants rejected versus mode144. Wave82 is recorded as an invalid unsupported-override attempt.

Wave85: fresh32 replication rejected wave84 tree-reach300 signal; mode144 retained.
