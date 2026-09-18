# Real-map controller candidates

All controllers accept immutable static geometry and ordinary native agent DTOs plus public time. Hidden predator state belongs only to the setup harness and evaluator. Infinite agent energy is a test assumption; native predator energy, rest and movement remain unchanged. Bait sites retain rear access and include boundary-wall gaps.

Current candidates are distinct experiments, not successive proven improvements:

| Module / class | Behavior | Evidence |
|---|---|---|
| `policy_v27:EmergencyGuides` | Terrain-aware guides; native emergency reserve births; surviving-guide release | Fresh4/4 under final causalv4; separate fresh16 extension pending |
| `policy_v30_wide_route:Guide` | Single guide; terrain-weighted routes prefer35/25-unit wall clearance, fallback11 | Fresh7/8 at300s; strongest completed single-guide fresh batch |
| `policy_v31_compact_fallback:Guide` | v30 plus compact/wider eligible geometry when ordinary selectors fail | Two fitted map repairs pass; no general availability or reliability estimate |
| `policy_v32_integrated:Guide` | Wide routes combined with moving final approach and viability choices | Known regressions; fresh8 independently scoring; not promoted |
| `policy_v33_adaptive_guard:Guide` | Adaptive endpoint safety floor | Known regressions; not promoted |
| `viability:Guide` | Conservative all-action three-step escape shield outside final intake | Fresh8 independently scoring; not promoted |

`spatial_geometry.py` preserves exact rectangle predicates while reducing candidate obstacle checks. `check_geometry.py` found zero mismatches in14,064 queries. The optimized burst replay agrees with the original native replay in all3001 actions/dynamics/decisions/observations after canonicalizing equal-distance Agent observation order; see the three preserved equivalence audits under `results/integrated_guide/`.

`run_batch.py` declares seeds and recursive policy dependency hashes before running cases, enforces resource limits, preserves failures and does not overwrite immutable receipts. It is for development batches. The separate frozen59-case runner is `research/reliability_eval/run_frozen_v4.py`; reserved seeds remain unused while candidates are still being selected.

Start with `research/simple_chase/SESSION3_RESULTS.md` for current outcomes and limitations. The viewer is http://127.0.0.1:9055/research/real_map_visualization/index.html. Native replays and image chunks stay local and are excluded from git because of their size.
