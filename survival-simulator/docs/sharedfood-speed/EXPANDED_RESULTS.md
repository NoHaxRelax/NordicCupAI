# Expanded local food: policy speed optimization

The game engine and policy parameters are unchanged. The optimized policy reuses temporary buffers, avoids rebuilding equivalent visibility sets, skips visibility work for objects already observed on the current tick, and reverses tree/fruit counting to visit each pair once.

## Controlled speed comparison

Same 100 seeds and SSH PC, with 12 workers for each frozen build. Every optimized game reproduced the original score, lifetime and number of simulation steps exactly.

| Build | Policy µs/tick | Engine µs/tick | Whole-loop CPU µs/tick | Wall seconds/game |
|---|---:|---:|---:|---:|
| Original | 693.8 | 537.9 | 1218.0 | 20.64 |
| Optimized | 574.4 | 529.0 | 1100.9 | 18.50 |

Policy time fell **17.2%** and whole-loop CPU time fell **9.6%**. Engine time is statistically noisy under concurrent load and was not optimized.

## Score check

The optimized build scored **1676.4** over 1,000 full games on the laptop; its pointwise bootstrap 95% interval is **1652.5–1700.6**. The earlier unchanged-policy Runpod panel scored **1676.4**. CPU families can produce different floating-point trajectories, so the 100-game same-host exact comparison is the clean regression check.

A tick is one update of the whole population. Timing includes concurrent-worker scheduling and should be compared within the same host/panel.
