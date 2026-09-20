# Candidate for under-20-second full games

Baseline f31f586 measured 25.68 seconds/game for sharing-only and 24.78 for congestion pricing on a fresh 32-map Runpod sample. See ../runpod32/RESULTS.md.

This candidate combines additional threshold/direction/endpoint lookup changes with an isolated snapshot of the ongoing `Speed up policy` task's allocation/visibility/counting changes from /home/Ucals/.codex/worktrees/sharedfood-speed. That live worktree was not modified. Its snapshot changes are incorporated here, not deployed to BO. No stats, horizons or behavioral hyperparameters were reduced for speed.

The extra endpoint index alone did not materially improve the 500-second laptop sample. Avoiding precise norms for rejected/unique wall matches did help; the combined candidate reduced that sample from 4.21 to 3.41 seconds. All six full-game action traces (109,229 ticks) match f31f586 exactly on seeds 19001, 19099, 19038 for both policies. Boundary checker passed. The fresh 100-map-per-policy Runpod benchmark is complete: sharing-only averages 17.57 seconds (95% CI 16.37–18.81), congestion pricing 17.56 seconds (16.44–18.71). Both are below 20 seconds on this sample at 32-worker load. Timing covers the native game loop, excluding initialization. All 64 overlapping full games match the prior version in score, lifetime and tick count. See ../runpod100/RESULTS.md.
