# Preflight checks

- New features disabled: expanded local food on seed15001 reproduces score1647.73150027206 and extinction1677.6999999995696 exactly; recorded in baseline-parity.json.
- Linux fork smoke on seeds17001 and17002 reproduces both baseline final scores and extinction times exactly; metadata and resumed results recorded.
- Native map tests: recover a rotated/translated two-rock frame; reject a single segment and ambiguous repeated shapes; another agent's predator observation triggers avoidance without changing the recipient's original observation list. Reproduce with `c++ -std=c++17 -O2 -I survival-simulator/fastsim survival-simulator/scripts/check_sharedfood20.cpp -o /tmp/sharedfood-check && /tmp/sharedfood-check`.
- `fastsim/check_boundary.py` passes: no engine or Python C-API access in the policy translation unit. Builds completed on all ten Runpod CPU pods.
- All20 seeded family configurations reached1100 simulated seconds on seed15001 without crashes. This exercises their900-second late switches. family-smoke.json records partial scores and wall times; these are implementation checks, not performance estimates or BO data.
- Shared-map computation is materially slower than the old own-view policy. Local four-worker checks took42–123 wall seconds per1100 simulated seconds, depending on family. Full evaluation will report actual per-tick timing; do not assume30seconds per game for these new mechanisms.

No policy settings were selected from these smoke scores. Training uses17001–17200, independent continuation validation18001–18200 and full-game testing19001–20000. Test seed15001 belongs to the already completed prior research panel.
