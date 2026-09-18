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
- 00:32 base1k (953 seeds): r21s0c2 2391 s / score 2553 vs best 2353 / 2492 (paired +38±14 s). Late-game harvest
  falls to 62-72% after 2000 s. diag1 (fruit fates, r21s0c2, 152 seeds): 80-90% of rotted late fruit had NO agent
  within 200 units during its life (55% none within 400) => coverage/discovery-limited, not supply-limited.
- 00:30 _npolicy.hpp 4be33b1c: late-game schedule (late_t + l_* overrides); regression identical when off.
  Running: s1 (32 perturbations, i/j/k/n1, seeds 2000-2127), p1 (12 single-param probes, n2), L1 (48 random late
  schedules, n3/n4/n5, seeds 2000-2095).
- 00:35 _npolicy.hpp: predator evasion layer (pred_mode=1: face nearest threat, back away, sprint <90) and
  merge_anchored (merge all wall-anchored families at once). pr0 (r21s0c2 with predators, evasion off vs on,
  64 seeds 5000-5063) running on n3/n4 next to L1. Engine facts for P2: predator 11 walk/15 sprint per tick,
  turn cap 0.3 rad/tick in direct chase; direct chase if agent faces away or is <90 away, otherwise a 45-degree
  pivot approach (closing ~10.6/tick); agents move then turn, turning uncapped (cost |turn|/2pi).
- 00:40 pr0 (with predators, 64 seeds): r21s0c2 no predator logic 846 s, 226 predator kills/run, score 684;
  evasion v0 963 s, 116 kills/run, score 982 (+282±50). Predators dominate: with them colonies die ~900-1000 s.
- 00:45 esc1 narrow escape grid (1 agent vs 1 fresh predator facing it, 30 s, energy 150/800 so no sprint):
  no evasion 79% killed; evasion v0 32%; kill rate by walking speed 10/15/20 about 60%/20%/8% (D>=80).
  => select for walking speed (heritable, cap 20). Added fit_speed_cap (fitness speed cap, default 1.5 => 15).
  Added no_spawn (tests only) and dbg_* engine hooks + nightsim/escape.py. sp1 (speed selection x predators) on n3.
- 00:55 P2 prep parked per Oscar (01:00: "optimise survival time now; the rest waits until after 2 am; narrow-gap
  baits only, never wall baits"). Parked code (all behind switches, default off): evasion+dodge+shared sightings,
  crevice site finder with median wall clustering (ts4: valid site in top-3 on 26/48 maps; every graded hold test
  kept the bait alive, predator closest ~19.5), bait role with replacement (untested). Perfect-trap model tv1:
  trapped within 100 s => +922 s; 30 s => +1167 s (the P2 prize).
- 00:55 c1 partial (~160 seeds): oracle tree knowledge +99±30 s (discovery headroom); merge_anchored 0;
  speed selection without predators -100 s; parameter candidates within noise.
- 01:00 P1 back in focus. q1 (32 probes of never-searched parameters: late breed reserve, tree-decay model n0/tree_half,
  emergency reserve, explore radius, births/tick, site_min, repost/switch/post/min_stay/watch_refresh, extra_old)
  on n2-n5 seeds 2000-2127 (r21s0c2 regression identical). c2 queued after c1 on i/j/k/n1 (seeds 10000-10511):
  oracle split (nearby-only 400/800, ages-only) + age_infer 20/40/60 (new: tree first seen in a cell viewed dt ago
  gets birth estimate at dt/2).
