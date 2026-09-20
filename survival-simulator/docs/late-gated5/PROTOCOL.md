# Late strategies with learned activation

Base: frozen expanded local food, the strongest established full-game baseline. No rock-face tuning result enters this experiment. Full game starts; normal energy, aging, predators and reproduction; horizon 3000 seconds or extinction. Public policy observations only. No SSH PC access.

Five combinations, selected using prior independent late-checkpoint gains and full-game results:

1. congestion_net: congestion pricing + positive net-energy fruit choices.
2. cooperative_auction: fruit auctions + congestion pricing.
3. renewal_relocation: returning to renewable food sites + leaving depleted areas.
4. capacity_budget: food-supported population capacity + reproduction energy budgeting.
5. auction_aging: fruit auctions + energy-efficiency preference among aging agents.

All include shared observations when active. Borrow only component-specific settings from the earlier winners; keep the base breeding-stat weights and predator policy. Before activation, use the exact base parameter profile. Switching back restores parameters, not erased observations or historical assignments; knowledge already learned remains. Shared map frames may stay merged after deactivation.

Each family has six hyperparameters: activation ticks 5000–24000 (500–2400 seconds); population threshold 3–24 with strict population < threshold; condition (time only, population only, both, either); persistence (reversible, latched forever after activation, one episode only); plus two component strengths. Population-only or OR conditions may activate early if the population is already low. Activation counts policy calls, starting at zero. Six dimensions, 16 trials (preset + five random + ten GP expected-improvement proposals), 100 full training seeds 41001–41100 reused for every trial/family. Normalize inputs; standardize scalar objectives for each GP fit, same fixed Matérn 5/2 kernel as previous runs. Maximize mean score subject to mean end-to-end worker game time <20 seconds, including initialization and result extraction, at up to32 workers per pod. No game is truncated for runtime.

A configuration at or above20 seconds on its training panel is ineligible. To steer the existing scalar GP away from it, its optimizer objective is -1000 -100*(seconds-20); feasible configurations use raw score. Preserve raw score and runtime separately. Freeze the highest-scoring eligible trial per family. If none qualifies, exclude that family. Any final mean runtime at or above20 means the strategy is not recommended, regardless of score.

Final: each eligible family winner plus unchanged base on the same fresh 2000 maps, seeds43001–45000. All models for one seed run on the same CPU, order rotated. No final score or time is used to retune. Report all evaluated variants, score95% intervals, paired score gains, train/test gaps, mean runtime and CPU microseconds per population tick, runtime eligibility, activation parameters and16-step BO evolution. Pointwise intervals, no multiplicity correction. Expected8000training+12000final=20000games in10000jobs if all five have eligible winners. Re-run prior-panel baseline neither required nor used for selection.

Use finished engine3ccd187, policy d6486e4 and exact sharing speed changes bc3069d. Verify gating state transitions and full-game equivalence at never/immediate activation, then a32-worker runtime preflight on32 fresh pilot seeds42001–42032 for BOTH default and always-active configurations of each family, plus baseline (352 full games). The coordinator refuses to start BO unless this preflight passes for every family. Optimize or omit any default family that misses20 seconds before BO. Keep existing rock-face job intact; start this workload only after its workers have finished, avoiding contention. Ten existing CPU pods, durable shared queue; leave pods running.
