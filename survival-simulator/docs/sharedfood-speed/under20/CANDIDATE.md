# Candidate for under-20-second full games

Baseline f31f586 measured 25.68 seconds/game for sharing-only and 24.78 for congestion pricing on a fresh 32-map Runpod sample. See ../runpod32/RESULTS.md.

This candidate combines additional threshold/direction/endpoint lookup changes with an isolated snapshot of the ongoing `Speed up policy` task's allocation/visibility/counting changes from /home/Ucals/.codex/worktrees/sharedfood-speed. That live worktree was not modified. Its snapshot changes are incorporated here, not deployed to BO. No stats, horizons or behavioral hyperparameters were reduced for speed.

The extra endpoint index alone did not materially improve the 500-second laptop sample. Avoiding precise norms for rejected/unique wall matches did help; the combined candidate reduced that sample from 4.21 to 3.41 seconds. All six full-game action traces (109,229 ticks) match f31f586 exactly on seeds 19001, 19099, 19038 for both policies. Boundary checker passed. A fresh 100-map-per-policy Runpod benchmark is pending; the under-20 target is not yet established.
