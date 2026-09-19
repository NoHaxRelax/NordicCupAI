# Best policy so far (no predators), 19 Sept 2026

| When | Policy (nightsim/_npolicy.hpp md5) | Config | Seeds | Mean survival | Score | Fruit |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 00:30 | 0f880eec (= peer's _orchard.hpp) | r21s0c2 (configs-all.json) | 453 (1000-1999, Linux EPYC) | 2376 | 2537 | 161 |
| 00:30 | 0f880eec | best (old best-config.json r3s2c6) | 453 | 2363 | 2503 | 140 |
| 00:32 | 0f880eec / 4be33b1c (late schedule off) | r21s0c2 | 953 (1000-1999) | 2391 | 2553 | 162 |
| 00:32 | same | best (r3s2c6) | 953 | 2353 | 2492 | 140 |

r21s0c2 vs best paired on 950 seeds: survival +38 ± 14 s, score +60 ± 14, fruit +22.6.

01:25: still r21s0c2 (configs-all.json). 1465 seeds overall: survival 2393 s, score 2555, fruit 162, 38 runs reach 3000 s.
No tested parameter or schedule beats it on 512 paired seeds (all within +-2 SE). Diagnostic headroom: perfect local
tree knowledge within 400 units +146 s.

## With predators (full games, 192 seeds 6000-6191, 19 Sept 03:20)
| Policy | Survival | Score | Eaten/game |
| --- | ---: | ---: | ---: |
| r21s0c2, no predator logic | 888 | 728 | 232 |
| + evasion v0 (flee within 200, shared sightings, dodge) | 923 | 976 | 60 |
| + flee only within 80, own sightings only (`e_r80_ns` in cfg-f11.json) | 1239 | 1260 | 130 |
Trap machinery (bait + guides) in full games: -200 s so far (see STATUS/GUIDE).
| + dodge radius 80, angle 1.4 (`pred_best_0342`, 03:42) | 1444 | 1462 | 139 |
| + flee within 70, face within 80 (`pred_best_0400`, confirmed on fresh seeds 7000-7191, 04:00) | 1507 | 1520 | 142 |
| `pred_best_0400` vs no predator logic, 576 fresh seeds 8000-8575 (04:25) | 1492 vs 887 (+605 ± 16) | 1502 vs 732 | 146 vs 225 |

Both best configurations (full parameter dicts) are in `best-configs.json`; pass either as the policy kwargs
(`nightsim/run.py --configs`, or `--kw` in the Python harnesses; the predator keys are ignored by the old Python
policy, which lacks the evasion layer).

06:00: policy code change kept: anchor() wall-side fix + boundary snap (nightsim/_npolicy.hpp). Paired regression on
the shipped configs: +13 ± 18 s (no predators, 512 seeds), +16 ± 16 s (with predators, 576 seeds). The same bug exists
in the Python policy (research/orchard/orchard.py, anchor candidate order); not ported there tonight.

## Final numbers with the fixed policy (fresh seeds 20000-20767, 768 each, 06:20 19 Sept)
| Config | Survival | Score | Fruit |
| --- | ---: | ---: | ---: |
| r21s0c2, no predators | 2432 | 2599 | 167 |
| pred_best_0400, with predators | 1533 | 1543 | 136 |

## 07:10 19 Sept: follow-up items
Best configs unchanged: r21s0c2 (no predators), pred_best_0400 (with predators). Relay guiding, sprint-lead,
wall-clear for guides, turn-rate escape, late-only trap, near-colony trap: none beats the shipped configs (see
results-summary.md). `pred_wallclear=1` on top of pred_best_0400 was +42 +- 27 s on 192 seeds but -22 +- 15 on 576 fresh seeds
(pooled 768: -6 +- 13): noise, not adopted. Shipped configs unchanged.

## 07:20 19 Sept: trap cost decomposition (f37, 192 seeds, vs pred_best_0400)
Baits alone -57 +- 27 s; baits + guides -140 +- 27; guides restricted to predators within 300 of the trap -158 +- 28.
No trap variant tested tonight is positive. Shipped configs unchanged.
