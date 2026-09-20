# Rock-face steering: 320-map retune

50 Bayesian iterations, 320 full games per iteration (seeds 21001–21320). Includes the previous 150 training maps plus 170 new maps. Five parameters and ranges unchanged. First trial is the previous training winner, then five random trials and 44 GP expected-improvement trials. Inputs normalized to [0,1]; targets standardized per fit. Fixed Matérn 5/2 kernel (length 0.4, noise 0.15). Objective is mean full-game score; no final-map feedback during tuning.

Freeze the best training configuration before evaluating the same 2,000 final maps, seeds 22001–24000. Run the new winner, previous rock-face winner, and unchanged expanded-food baseline on each seed on the same CPU. This is a reused benchmark, not a fresh holdout: the strategy family was chosen after inspecting the prior final results. Report pointwise paired bootstrap intervals, training evolution, score intervals, runtime and CPU per population tick.

16,000 training games + 6,000 comparison games = 22,000 games (18,000 queue jobs). Separate pilot and correctness checks do not select parameters. Full energy costs, aging and predators, horizon 3000 seconds or extinction, policy seed 0. No privileged policy inputs. Reuse existing ten CPU pods with 32 workers each; leave pods running. Do not access SSH PC.

Integrated policy optimization from d6486e4 and a frozen engine-speed snapshot; source provenance and exactness checks saved here before launch. Build on each host, preserving floating-point safeguards. One shared durable queue; one BO iteration waits for all 320 results. Restart the coordinator with the same output directory to resume. Never alter binaries/config spaces during a run.
