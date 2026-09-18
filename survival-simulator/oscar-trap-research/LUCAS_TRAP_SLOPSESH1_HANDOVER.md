# Lucas trap session 1 handover — 17 September 2026

Branch: `survival-simulator/lucas-trap-slopsesh1`.

The arranged narrow-gap trap can acquire and retain **33 predators for a full 3,000-second game** with the bait 5 units inside the mouth. **We have not successfully delivered a predator from a random real-map encounter to that trap.** Capacity and retention are demonstrated under arranged conditions; a reliable deployable strategy is unfinished.

This handover supplements the original [HANDOFF.md](HANDOFF.md), rather than replacing its earlier findings. Research stopped to package this branch at the user's request. No competition endpoint was called and no score was submitted. The vendored simulator is unchanged.

## Constraints to preserve

- Keep bait depth at **5 units**; the user considers depth sufficiently tested.
- Refill agent energy indefinitely for now. Predator energy, resting and movement remain native.
- Guide sacrifices are allowed. The goal is ultimately 33 predators retained throughout a game.
- Perfect **static map** information is currently allowed. After the first native predator observation, the policy must use ordinary observations, public time and static geometry only.
- The intended scenario is a fruit gatherer encountering a predator, spawning a child and guiding the predator to a dedicated bait. Finding predators is not the research bottleneck to optimize yet.
- In the latest harness, the setup agent approaches the predator using an explicitly labeled oracle until the first ordinary Predator DTO. `oracle_handoff()` resets setup-induced navigation belief before the first policy action after that cutoff. This is a diagnostic setup, not an observation-only search demonstration.
- Preserve every frame for new visual tests using the native renderer. Do not represent a reproduction as original footage.
- The user requested Sol subagents and a usage stop at 60% remaining, replacing the previous 80% threshold. Experiments have finished; do not automatically launch more on resumption without checking the current task and budget.

## Results worth keeping

| Test | Result | Practical limit |
| --- | --- | --- |
| Depth 5, gap 15, seed 483, guided arrivals every 90 s | 33/33 acquired and retained for 3,000 s; no losses; bait alive; 33 guides sacrificed | Arranged geometry and arrivals, not real-map routing |
| Depth 5 direct arrivals, gap 19/seed 641 and gap 11/seed 642 | Both retained 33/33 for 3,000 s | Direct-arrival controls |
| Depth 5 crowd tests, gaps 11/15/19, two seeds each | All six survived 120 s with 33 predators | Short horizon |
| Depth 4.9, gap 19 crowd tests | Both failed | Single-predator survival does not establish crowd safety |
| Earlier depth 30, v4 long-lead HOLD75, seeds 4331/483/484 | Three full games retained 33/33 | Superseded depth choice; arranged routing |
| Real-map delivery experiments | **Zero demonstrated deliveries** | Neither mass accumulation nor whole-game retention after delivery established |

The depth-5 full guided replay ends `77cfe59d`; direct controls end `89e44289` and `dc8f9988`. The earlier interrupted guided run `94db2c0c` held 23/23 through 2,053.6 seconds when the former usage stop ended it: it is a partial run, not a full-game success. In the crowd tests the closest observed center distance was 15.016738; capture requires strictly less than 15 (predator radius 10 plus agent radius 5). This is a small geometric margin.

Depths -5, 0, 2.5 and 4 failed single-predator tests across the three tested gaps; 4.5 survived only the narrow gap 11. No more depth sweeps are needed now. A separate single-wall alternative held three predators in ten tests, including four full games; “11 such traps could hold 33” remains arithmetic, not a tested placement-and-delivery strategy.

See [wall funneling](docs/survival-wall-funneling.md), [sequential intake](docs/predator-sequential-intake.md), and the source/receipt directories under `survival/research` and `survival/results`.

## How many real maps have a bait spot?

The static census generated **128 native maps**, seeds 10000–10127, without simulation steps or actors. It saved exact map geometry and candidate sites in [real_map_census](survival/results/real_map_census/summary.json).

- Current conservative selector: gap 10.9–19.1, overlap at least 54.9, free depth-5 bait position and clear staging from 125 to 75 units outside the mouth: **58/128 (45.3%)**, Wilson 95% interval 37.0–53.9%.
- Also requiring a clear approach to the first blocking face: **53/128 (41.4%)**, interval 33.2–50.1%.
- Thus **roughly 40–45%** is the useful current estimate for eligible geometry, **not the probability of successful delivery**.
- Relaxing minimum overlap to 20 gives 127/128 candidates with clear approach; overlap 30 gives 121/128 and 40 gives 109/128. Those shorter overlaps have not been validated for delivery and retention.
- Interior refuge geometry alone appears in 74/128 maps. Including arena boundaries gives 118/128, but many candidates lack usable staging or entrances. These broader counts should not be presented as validated bait availability.

An early optional corridor calculation accidentally ended inside an expanded wall and was corrected; the 58/128 staging count was unaffected. See [census README](survival/research/real_map_census/README.md) for definitions and reproduction.

## Real-map progress and the current blocker

The harness uses the native 1600×1200 world, obstacles, terrain, fruit, trees and future predator spawning. Bait, parent and predator start at seeded random positions and headings; the bait must navigate to its site. The native birth creates the child nearby. Agent energy alone is refilled. Native DTOs are JSON-roundtripped at the controller boundary. Hidden creature position, targets and rest flags are used only in evaluation after the encounter cutoff.

The latest frozen controller is [policy_v15_hearing_reacquisition_frozen.py](survival/research/real_map_guide_sol/policy_v15_hearing_reacquisition_frozen.py), SHA-256 `4455972445d877117433907a9fb855a9112520fc0ee86c3d9afd0bbc7a72e8ab`; `policy.py` currently matches it. Snapshots v1–v15 are retained. Read the [policy README](survival/research/real_map_guide_sol/README.md) and [final intake report](survival/research/real_map_intake_sol/REPORT.md) before changing it.

Implemented improvements include brief look-backs every four ticks while translating, looking away between checks, pursuit confirmation from observed world motion and gaze history, revoking stale confirmation, gating progress when pursuit is unconfirmed, native-child role assignment, and static route/corner guards. Earlier bugs included false arrival on an empty path, ambiguous parallel-wall localization, setup actions corrupting dead reckoning, impossible escape separation thresholds, corner cutting and repeated direction changes in concave corners.

**Critical timing detail:** native agent observations are cached before predators move. Even a fresh DTO is one predator movement stale. An escape safety calculation must allow the already-unseen movement, the next movement and capture radius, with margin. One failure had a DTO distance of 33.49 but an actual distance of 24.70 at the death action.

The final qualifying v15 run is receipt/replay suffix **922a9679** (map 5101, fixture 9203). Bait deployed at 1.2 s, encounter and birth occurred at 4.4 s, and all agents survived 60 s. It had **zero physical entries and zero deliveries**; the predator finished 1,520.9 units from the mouth. The intended 54–58-unit reacquisition orbit was physically blocked by obstacles. Measured distances at predator wake were 69.2, 73.6 and 74.1, beyond native hearing range 60. The guide became pinned around [1490,1125], with the predator around [1557,1156]. Survival is not evidence of successful guiding.

**Most useful next experiment:** find a physically reachable standoff within hearing range using static radius-5 path clearance, while retaining a latency-aware escape margin and observation-based pursuit confirmation. Test delivery of one predator before adding more. Do not tune the bait depth again. Any use of hidden predator rest or target to control the guide would invalidate the observation-only claim.

Most iterations reused map 5101 with fixtures 9101/9102/9201/9202/9203. A held-out map 10000 test also failed reacquisition and delivery. Map 5102 had no site under the conservative selector and has an unsupported-setup receipt, not a simulated replay. Repeated versions on the same seeds are not independent reliability samples.

## Evidence accounting and limitations

- The final real-map viewer contains **23 entries and 15,773 native frames**: 13 delivery failures, four no-birth diagnostics, three excluded no-encounter runs, two short pilots and one unsupported setup. There are zero successful deliveries. Categories include reproductions and should not be converted into a statistical success estimate.
- No-birth wrappers deliberately suppress spawning to isolate single-guide routing. They are excluded from the qualifying parent/child denominator. The latest, `d52e175f`, kept the bait and guide alive for 60 s but delivered nothing; final predator distance was 234.53 units. Internal birth-intent events in wrappers do not mean a native child was born.
- Original receipts are immutable. `*.correction.json` sidecars fix interpretation and scoring labels, and record receipt/source hashes. Read the receipt together with its sidecar. The viewer merges these and labels corrected scores.
- Two early state-only pilots have separately labeled native reproductions: `ec135f67` → `6cecd241`, and `7cf5b851` → `7cdbcc20`. The reproduction frames are not the original pilot footage.
- The current intake harness hash is `bf52f369d60d09cee01c05f68aa0d322fff5b3cc2fe9a8ccd9b40e0c075dfd64`, archived as `archive/run_bf52f369.py`. Four older referenced harness hashes are not archived yet: `2645d88b`, `6a2c2a7f`, `9d9ed9b6`, `ca14e0ad`. Local tool history may allow reconstruction; provenance is not complete for those older runs.
- Food/energy sustainability, actual gatherer behavior, observation-only map discovery, reliable child recruitment, multiple simultaneous deliveries and real-map whole-game retention remain unproven.

## Viewing and resuming

From `survival-simulator/oscar-trap-research/survival`, using a Python environment with the vendored simulator requirements installed:

```sh
python debugger/serve.py --port 9053
# In another terminal:
python research/strategy_visualization/serve_visualizer.py --port 9055
```

- Overview and independent Ask Sol helper: http://127.0.0.1:9055/research/strategy_visualization/index.html
- Real-map every-frame viewer: http://127.0.0.1:9055/research/real_map_visualization/index.html
- Depth every-frame viewer: http://127.0.0.1:9055/research/depth5_test/viz.html
- Complete local replay library: http://127.0.0.1:9053/

The Ask Sol helper uses the local Codex CLI (`CODEX_BIN` or `codex`), a separate read-only thread and local authentication. Its thread file is `/tmp/predator-strategy-sol-helper-thread`. Usage telemetry configuration is session-specific and must be adapted on another machine; the helper does not give a fresh clone a shared remote agent. Do not commit credentials or chat/session logs.

**Git does not contain the original recordings or gzip frame chunks.** This follows the existing repository ignore policy. The checksummed [LOCAL_REPLAY_INVENTORY.json](LOCAL_REPLAY_INVENTORY.json) lists original local recordings, sizes and SHA-256 hashes; it is not a download service. They remain under `/home/Ucals/projects/NordicCupAI-wall-funneling/survival-simulator/oscar-trap-research/survival/results/`. A fresh clone has code, receipts, manifests and small overview assets, but the all-frame viewers need the recordings/chunks transferred separately or new experiments run. Restore original files at their listed relative paths, then regenerate chunks:

```sh
python research/real_map_visualization/build_manifest.py
python research/depth5_test/build_manifest.py
```

Both builders validate native images, contiguous 0.1-second timestamps and frame totals, then produce 100-frame chunks with a shorter final chunk. The depth viewer was verified at 46 cases / 132,588 frames. The real-map viewer received desktop browser verification with accurate role and failure labels. The intake harness refreshes its viewer best-effort after saving; a viewer failure must not invalidate an already saved recording.

Use the existing experiment READMEs for exact commands and frozen policies. Do not rerun historical sources and overwrite receipts. Save each new run under a unique ID, record every frame, and check discovery in the live replay catalog. The unrelated pre-existing `survival/results/offset_dip/` directory is intentionally outside this branch's new research snapshot.

## Packaging verification

All newly included research Python files passed syntax compilation. The live Survival Lab catalog contains both final v15 recordings (`922a9679` and `d52e175f`). The inventory covers 385 original recordings totaling 5,002,478,560 bytes. Git staging contains no vendored simulator edits, recordings, compiled Python, session logs or unrelated `offset_dip` files. `git diff --check` reports only eight extra blank lines at EOF in research sources, including hashed frozen/source-provenance files; these bytes were preserved rather than changing the recorded source hashes. No additional simulations were run merely to package the branch.
