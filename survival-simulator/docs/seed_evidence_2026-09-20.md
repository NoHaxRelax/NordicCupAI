# Seed evidence from spawn positions and shared geometry

Implemented 20 September 2026 in [seed_survey_evidence.py](../scripts/seed_survey_evidence.py), integrated into [survey collection](../scripts/seed_survey_probe.py) and [the Rust search runner](../scripts/search_survey_rust.py).

## Behavior

Surveys now retain every raw public frame, its canonical hash, the actual action history, and a final shared-world snapshot. Evidence can be rebuilt later without the target seed, simulator state or localization audit. The first frame must be the native first empty tick at 0.1 seconds, with every intervening 0.1-second frame retained. A recording that starts after walking cannot silently label its first observed positions as spawn positions.

An offline shared geometry reconstruction joins sightings across agents and time. It determines absolute headings from directed edges and known-size boundary walls, then registers pairs of complete static rock edges. ID-bearing agent sightings and stationary turns add connections. Boundary observations anchor connected components to world coordinates. A later anchor can therefore locate an earlier observation, including a founder's spawn, without reversing commanded movement or assuming collisions followed the intended route.

The shared model's existing `origin` is retained in the snapshot for inspection, but is not trusted as an exact constraint: current-pose corrections do not back-correct its history. The new reconstruction uses the original observations directly and leaves unsupported origins unresolved.

| Evidence | Collection and use |
| --- | --- |
| Founder positions | Agent ID, reconstructed first position, initial heading and positional tolerance; compared with the candidate's initial agents. |
| Founder distances | Pairwise distances within consistent connected components; usable even before absolute boundary alignment. |
| Rock rectangles | Width/height paired from perpendicular edges sharing a corner; coordinates added when the observation is localized. A candidate must match both dimensions on the same rock. |
| Rock edges | Whole endpoints in world coordinates, with positional tolerance; candidate edges must match both endpoints. |
| Biome transitions | Consecutive observations on opposite sides of a label change, with times and positions. Half of the 64-sample scan budget is available for short transition brackets, with the rest selected for spatial coverage and label diversity. |
| River observations | Retained and checked against generated candidate maps; excluded from the Rust land-prefix scan because rivers overwrite the base Voronoi map. |
| Initial trees | Raw distance/angle vectors from the first frame, associated with founder IDs. Checked at each candidate's own spawn and heading, even when the target spawn is unresolved. Later trees remain in raw history and replay; they are not treated as initial trees. |

All other public fields, including agent traits and dynamic sightings, remain covered by the final action replay. Missing observations are not treated as proof of absence.

## Search stages and compatibility

The existing parallel Rust scanner consumes the stronger biome samples automatically. It still enumerates the selected unsigned 32-bit interval. Candidate generation and the new geometry checks run in Python before full public-history replay, since rocks and founders appear after expensive RNG-consuming terrain generation. This change does not claim an algebraic RNG inverse or a proportional speedup for scanning the full seed space.

Schema version 2 records contain `public_frames` and `seed_evidence`. Before native verification, the runner validates the raw-frame hashes and reconstructs the evidence to detect stale or edited constraints. Each verification result reports the new match flags and constraint counts. Original hash-only surveys still work with their original checks; information cannot be reconstructed from hashes alone. Candidate truncation and native-world caps continue to prevent an unchecked result from being certified unique.

## Run it

From `survival-simulator`, collect a local test world and search its saved observations:

```powershell
python scripts/seed_survey_probe.py --targets 78431 --seconds 30 --collect-only --output runs/survey.json
python scripts/search_survey_rust.py --input runs/survey.json --count 10000000 --threads 20 --verify --output runs/search.json
```

Choose new output filenames. `--targets` selects the simulated test world used to produce public observations; its value is not supplied to the evidence reconstruction or used to restrict candidate seeds. For live observations, `extract_evidence(public_frames, action_log)` uses the same public-only interface.

Rebuild an archived survey after changing extraction code, without running its target simulator:

```powershell
python scripts/seed_survey_evidence.py --input runs/survey.json --output runs/rebuilt-survey.json
```

Rebuilding replaces derived constraints and biome samples, clears obsolete search/verification results, and preserves original public frames and hashes. The raw history makes files larger: the measured 300-frame survey is about 7.9 MB as indented JSON. Working survey archives stay under ignored `runs/`; collect a new archive with the command above when reproducing from a clean checkout.

## Validation

Thirteen new tests cover later anchoring after a walk whose actual displacement differs from the command, cross-agent registration, stationary rotation, unresolved groups, repeated ambiguous geometry, conflicting anchors, stale observations, spawn distances before anchoring, late/incomplete recordings, paired-dimension rejection, positional rejection, tree timing, biome transitions, raw hashes and audit independence. The native integration test within this suite collects fresh 30-second surveys for seeds 3, 11 and 20260919, checks the Rust filter, and verifies all recorded public frames and every new constraint category. All passed. The five existing integration tests also passed, preserving original saved-survey results and verification caps.

An additional fresh seed-78431 survey recovered three spawn positions by 0.7 seconds, four by 10 seconds and all five by 20 seconds. At 30 seconds it retained five spawn positions, ten spawn distances, 48 rectangle constraints (42 with positions, 45 distinct width/height pairs), 138 positioned rock edges, one initial tree sighting, 1,036 localized biome observations and ten biome transitions. These are correlated observations and constraint counts, not independent information bits or 138 different rocks.

The checked-in [benchmark result](seed_evidence_benchmark_2026-09-20.json) records one-million/single-thread and ten-million/20-thread scans, input/source hashes, all constraint checks and final replay. On the local Intel Core i7-13800H, one million seeds took 7.22 seconds on one thread; ten million took **6.22 seconds on 20 threads**. The latter retained only seed 78431, which passed every new constraint and all 300 public frames. Total runner time was **14.47 seconds**, including evidence validation, imports, candidate generation and replay, with collection already complete. These are individual measurements rather than a controlled before/after speedup claim.

The raw archive used locally is `runs/seed-evidence-20260920-rebuilt.json`. No new Runpod instance was provisioned for this change. The existing Runpod launch script now runs the new tests too.

## Limits

Precise edge-pair constellations are used as landmark identifiers. Quantized keys only propose matches; unrounded descriptors must agree. Signatures repeated at distinct locations within a frame are excluded, inconsistent components are discarded, and cached sightings are not reused as fresh geometry. Identical repeated constellations never observed together can still be ambiguous; this is a source-specific identification assumption, not a mathematical uniqueness proof. Final replay remains required.

Registration tolerances grow along observation links; excessively long/uncertain chains are not used for absolute positions. Biome prefix samples additionally include a square-root-of-two margin for the native integer-pixel lookup. Existing shared-estimator samples can supplement coverage with their original heuristic radii. Tests validate the supported generator and maps; they do not establish universal confidence bounds for changed APIs or rounded coordinates.

The native 1600 by 1200 map, 0.1-second tick, full unrounded directed edge protocol, initial empty action step and unsigned 32-bit seed range remain assumptions. Exact public-frame hashes remain platform-sensitive, as documented in the earlier Runpod report.
