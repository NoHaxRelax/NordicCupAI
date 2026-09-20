# Self-stuck predators: five full-game BO families

Branch `codex/selfstuck-bo5`, parent `fba2549`. Native engine unchanged. Baseline is the frozen expanded-local-food configuration from localfood20 pod 7. Orchard and reproduction settings remain fixed, including existing map/fruit sharing; the new expensive shared-food layer is disabled.

## Hypotheses

1. **boundary_steer**: use an observed map boundary to aim pursuit toward it, then leave detection sideways.
2. **rock_face_steer**: aim pursuit toward the nearest observed rock face, then disengage.
3. **rock_corner_escape**: aim toward an observed rock endpoint and reward escape behind the edge.
4. **preserve_stationary**: infer stationary predators from consecutive own sightings, then avoid disturbing them or entering their detection area.
5. **adaptive_stuck**: combine boundary/rock-face steering with preservation of observed stationary predators.

These are heuristic hypotheses, not proven trap geometry. No bait agents, infinite energy, census locations or hidden predator state are supplied. No changes to fruit production, movement, predation or engine rules. Public relative sightings, heading reports, biome, energy, odometry and observed edges are the only decision inputs. Tracking has no true predator IDs; it can confuse nearby predators. Safety uses a conservative 15-unit one-step pursuit bound and known-wall collision screening, not a privileged engine rollout. It cannot guarantee escape. Lost sightings and coordinate jumps reset the stationary inference.

Six BO parameters: encounter radius 100–240; desired distance 75–140; geometric reward 10–150; release angle 5–40 degrees; stationary evidence duration 5–60 seconds; minimum energy fraction for steering 0.2–0.75. Only active parameters are tuned: five for the three steering families, three for preservation, six for the combined family. Baseline is exact when stuck_mode=0. Controller is in models/self_stuck.hpp and compiles inside the observation-only policy unit. The boundary checker includes that file in its full checks.

## Prespecified evaluation

- Pilot: 16 maps, seeds 25001–25016, each family default and baseline. Used for runtime/functionality only, not parameter selection.
- Training: **50 trials per family × 150 full games**, seeds **21001–21150** reused across all trials/families, 37,500 games. Policy seed 0; normal predators/energy/aging; horizon 3000 seconds or extinction. Objective: mean score from game start.
- BO: initial default, five random startup trials, 44 Matérn 5/2 GP expected-improvement proposals. Inputs normalized to [0,1]; scalar targets centered/scaled per fit. GP uses fixed length scale 0.4 and noise 0.15 in normalized target units, not marginal-likelihood hyperparameter fitting. Score is the sole optimization metric; compute is reported separately.
- Freeze the argmax training configuration separately for each family before final jobs are queued. No test feedback used to tune/select parameters.
- Final: **all five frozen family winners plus baseline × the same 2,000 fresh seeds 22001–24000**, 12,000 games. Report all means and 95% bootstrap CIs, paired differences versus baseline, train versus test gap and mean compute time per population tick. Pointwise CIs are not multiplicity-adjusted.
- Total: **49,500 campaign games**, plus 96 pilot games.

## Scheduling and reproducibility

A durable local coordinator maintains one queue shared by ten existing Runpod CPU pods, up to 32 native game processes each. It dispatches another game immediately when a worker completes. Each family's next BO trial depends only on its own 150 results. Idle capacity can help any family. Completed game records and every trial configuration are persisted; same-output-directory restart recovers completed jobs. Source hashes are checked against each worker's hello. Disconnects requeue unfinished jobs; completion IDs deduplicate results. Game exceptions halt the campaign.

Final evaluation bundles all six policies for a seed onto one worker to make each paired comparison use the same CPU and software. Model execution order rotates by seed. Training games use the shared pool; CPU floating-point differences may add training noise. Per-pod source/build provenance is retained. A local coordinator shutdown stops dispatch; restart with the same command recovers progress.

CPU pods must remain running after completion. This is offline research, not an official validation submission.
