# Overnight results (19 Sept 2026), survival simulator, C++ native engine + policy on Runpod

All numbers: nightsim fork (survival/nightsim = native engine + orchard policy in C++, bit-identical to the
Python policy), Linux EPYC pods, 3000 s horizon, paired comparisons on identical seeds. No competition API used.

## Phase 1: no predators (teammates handle predators)

| Measurement | Seeds | Result |
| --- | ---: | --- |
| r21s0c2 vs old best (r3s2c6) | 950 | survival 2391 vs 2353 s (+38 ± 14), score 2553 vs 2492 (+60 ± 14), fruit +22.6 |
| Late-game harvest (r21s0c2) | 450 | 92% of spawned fruit eaten before 1000 s, 62-72% after 2000 s |
| Rotted late fruit by nearest agent (r21s0c2) | 152 | 80-90% had no agent within 200 units during its life; 55% none within 400 |
| Perfect tree knowledge (oracle, diagnostic) | ~500 | +99 ± 30 s: the discovery headroom |
| 32 perturbations, 12 single-parameter probes, 48 late-game schedules | 96-128 each | none clearly better; late-game changes lean negative |

Survival correlates weakly with the map's tree count (+0.1) and more with the colony's own state at 1500 s
(agents alive +0.38, fruit eaten 1500-2000 s +0.45).

## Phase 2 preview (parked until 02:00 per Oscar)

| Measurement | Seeds | Result |
| --- | ---: | --- |
| r21s0c2 with predators, no predator logic | 64 | 846 s, 226 agents eaten per game |
| + evasion (face, back away, sprint < 90) | 64 | 963 s, 116 eaten (score +282 ± 50) |
| Perfect-trap model: every predator trapped T s after spawn | 53 each | T=600: +205 s; 300: +481; 100: +922; 30: +1167 |
| One agent vs one fresh predator, 30 s (walk speed 10/15/20) | 288 per variant | caught ~60%/20%/8% with evasion; 79% without |
