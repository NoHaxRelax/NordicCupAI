# Training evolution versus full-game gains

Training gain is additional score relative to unchanged scheduled breeding on the same 200 checkpoints (baseline continuation mean: 242.8). Full-game gain compares with scheduled breeding on 1000 fresh paired maps (baseline mean: 1512.5). Each evolution cell is the best continuation score so far at trials 1 / 8 / 16 / 24 / 32.

| Family | Best continuation score: 1 / 8 / 16 / 24 / 32 | Training gain | Full-game gain |
|---|---|---:|---:|
| birth_pacing | 292.2 → 368.6 → 393.9 → 393.9 → 394.5 | +151.8 | +31.1 |
| breed_feeding | 394.2 → 394.2 → 394.2 → 394.2 → 394.2 | +151.4 | +2.5 |
| cluster_food | 361.4 → 396.3 → 396.3 → 403.8 → 427.8 | +185.0 | +50.0 |
| conserve_explore | 242.8 → 402.3 → 402.3 → 402.3 → 402.3 | +159.5 | +18.9 |
| early_heirs | 242.8 → 382.1 → 417.8 → 417.8 → 417.8 | +175.0 | +16.8 |
| economy_mix | 242.8 → 401.9 → 406.0 → 406.0 → 406.0 | +163.3 | +47.3 |
| elite_heirs | 242.8 → 346.3 → 374.8 → 404.5 → 405.5 | +162.8 | +43.1 |
| late_gaze | 379.5 → 398.7 → 398.7 → 398.7 → 398.7 | +155.9 | +25.4 |
| late_reserves | 242.8 → 373.0 → 377.5 → 380.2 → 393.8 | +151.1 | +36.3 |
| local_food | 242.8 → 386.6 → 428.4 → 428.4 → 435.4 | +192.6 | +85.7 |
| nursery | 403.7 → 403.7 → 406.3 → 410.7 → 410.7 | +167.9 | +34.5 |
| old_priority | 243.5 → 405.1 → 414.9 → 414.9 → 420.3 | +177.6 | +16.1 |
| relocate | 242.8 → 395.7 → 405.1 → 405.1 → 405.1 | +162.3 | +46.2 |
| retirement | 242.8 → 410.7 → 435.1 → 435.1 → 435.1 | +192.4 | +9.2 |
| ripe_food | 242.8 → 372.7 → 401.8 → 401.8 → 401.8 | +159.1 | -18.3 |
| scan | 242.8 → 392.3 → 407.7 → 407.7 → 407.7 | +164.9 | +35.9 |
| small_population | 292.2 → 403.5 → 403.5 → 403.5 → 409.0 | +166.3 | -107.0 |
| spread_food | 388.6 → 402.4 → 402.9 → 402.9 → 402.9 | +160.2 | +36.6 |
| tree_capacity | 242.8 → 383.7 → 387.1 → 388.0 → 388.4 | +145.7 | -61.6 |
| wide_food | 242.8 → 370.4 → 370.4 → 408.3 → 408.3 | +165.6 | +36.2 |

Training scores are selected maxima and are optimistic. Their gains do not imply the same full-game benefit. For confidence intervals and paired tests see [final report](final/RESULTS.md).

## Three sample replays

Seeds 12001, 12002, 12003 were chosen as the first three evaluation seeds, before viewing results. All use local_food and record every 0.1-second tick, including initial and terminal states. Each recording matches an independent uninterrupted full native run on the recording machine. All three final scores also exactly match the saved pod benchmark rows: 1351.446822248349, 1944.9971069481599 and 1713.152223273618.

Run `python scripts/late250_replay.py record --seed 12001 --folder /tmp/late250-replays/seed-12001` (repeat for 12002 and 12003), then `python scripts/late250_replay.py serve --folder /tmp/late250-replays --port 9097`. Open http://localhost:9097. Requires the compiled native modules, pygame, scipy, pydantic and numpy. Local recordings are not checked into Git.
