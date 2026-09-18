# Overnight status (19 Sept 2026) — read this first on every tick

Rules (Oscar, 23:55): C++ only (nightsim fork: survival/nightsim, engine+policy natively), runs ONLY on Runpod pods
(never simulate on the laptop; compiling locally is fine), NEVER touch the competition API, push only to
survival-simulator/oscar-* branches, keep compact data (no replays), always keep the best policy saved.
Budget: $100 new spend from 23:55 CEST, up to 8 pods ($7.68/h => ~13 h). Pods: pods.txt; driver: night.sh.

Phases
- P1 (to ~02:00): no-predator survival in C++ (nightsim/_npolicy.hpp + configs). Baseline r21s0c2.
- P2 (02:00 on): Lucas's latest trap/bait work (report: lucas-trap-report.md) -> port to C++, combine with
  survival policy; predator regime tests; many-predator late game; turn-rate escape; old agents as bait/guides.

Best so far (no predators): see BEST.md (policy hash + config + score).

Log (newest last)
- 00:05 5 new pods n1..n5 (EU-RO-1, cpu3c-32, $0.96/h each) + i/j/k; cpu-a stopped (never terminate).
- 00:15 nightsim fork built on all 8 pods (policy md5 0f880eec = peer's _orchard.hpp).
- 00:25 base1k running on all 8 pods (launcher bug: 'cd &&' kept ssh open; fixed with ';').
