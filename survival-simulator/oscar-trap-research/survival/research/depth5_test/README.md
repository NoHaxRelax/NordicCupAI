# Bait depth boundary

Depth means the bait center's distance **inside** the gap mouth. Negative
depth is outside. These experiments use the unmodified simulator, refill
agent energy, and preserve native predator energy, rest, and movement.
Controllers use native observations; these are arranged approach fixtures,
not random-start real-map delivery tests.

## Completed short tests

The geometry has unequal 90/70-unit obstacle faces, a -5 face offset, and
gap widths 11, 15, or 19. Every simulation step (0.1 seconds) is recorded
with the native renderer, including failed cases.

Single-predator tests ran for 60 seconds unless the bait died:

| Bait depth | Gap 11 | Gap 15 | Gap 19 |
| --- | --- | --- | --- |
| -5, 0, 2.5 | Died | Died | Died |
| 4 | Died | Died | Died |
| 4.5 | Survived | Died | Died |
| 4.9 | Survived | Survived | Survived |
| 5 | Survived | Survived | Survived |
| 30 | Survived | Survived | Survived |

Crowd tests introduced 33 predators at 0.1-second intervals, with lateral
spread 15 and heading jitter 0.3, and ran for 120 seconds. Two seeds
(631 and 632) were tested per configuration:

| Bait depth | Gap 11 | Gap 15 | Gap 19 |
| --- | --- | --- | --- |
| 4.5 | 2/2 held all 33 | 0/2 bait survived | 0/2 bait survived |
| 4.9 | 2/2 held all 33 | 2/2 held all 33 | 0/2 bait survived |
| 5 | 2/2 held all 33 | 2/2 held all 33 | 2/2 held all 33 |

Thus 5 is the smallest tested depth that passed every crowd case. This is
an empirical boundary for these fixtures, not a universal safety guarantee.
The 4.9 failures show why a single approaching predator is insufficient.
Native contact kills at center distance strictly below 15 (radii 10 + 5).

## Long tests

A guided depth-5 run was interrupted at 2,053.6 seconds by the former
80%-remaining quota stop. It had acquired and retained all 23 arrivals,
with no losses, a living bait, and 23 sacrificed guides. It is partial
evidence only. The full 3,000-second rerun passed: all 33 acquired, zero
physical or joint losses, all 33 held over the final 30 seconds, bait alive,
and all 33 guides sacrificed. Maximum distance from the mouth after
acquisition was 29.323; no continuous active target switch was observed.
Seed 483, gap 15, source hash
`f7f3e4a939cb52003f4207533b2f3e6494238129336d58dc58ee35ef66510db1`.
Replay suffix `77cfe59d`; 30,001 native frames.
Gap-11/gap-19 direct-arrival full-game controls both passed 33/33 with
zero losses over 3,000 seconds (seeds 642/641). The user has fixed depth
at 5 and ended further depth experiments. None of these arranged fixtures demonstrates
independent random-start delivery across a native map.

## Inspect the evidence

- `results/depth5_test/frontier-summary.json`: single-predator depth sweep.
- `results/depth5_test/crowd-frontier-summary.json`: all 18 crowd cases.
- `results/depth5_test/clearances.json`: initial 5-versus-30 clearances.
- `results/depth5_test/guided/`: guided run receipts and replays.
- [Every-frame viewer](http://127.0.0.1:9055/research/depth5_test/viz.html)
- [Full native replay library](http://127.0.0.1:9053/)

`build_manifest.py` checks frame count, every timestamp, and native images
before publishing 100-frame chunks for the viewer. No rendered frames are
discarded. The original replay remains the complete state/action record.
