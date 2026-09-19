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
