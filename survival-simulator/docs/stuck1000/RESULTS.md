# Expanded local food: predator confinement census

All 1,000 games completed on seeds 19001–20000 using the frozen expanded-local-food policy. Spectator tracking did not alter observations or actions. Every selected replay reproduced the original game's score and end time exactly.

Screening called an episode when one predator remained inside a 40×40-unit axis-aligned box for more than 60 simulated seconds, sampled once per second. It found:

- 864 of 1,000 maps with at least one candidate episode.
- 4,902 episodes total; median 398.05 seconds, mean 535.80 seconds, maximum 2,705.30 seconds.
- 2,944 episodes lasting at least 300 seconds and 788 lasting at least 1,000 seconds.
- 2,410 episodes adjacent to the map boundary (within 60 units), across 623 maps.

This is a confinement screen, not yet a reachability or lure-success result. It can include resting, tight orbits, pursuit behavior, and separate episodes from the same location. The ten displayed cases were rerun at 0.1-second resolution. All ten remained inside the stated 40-unit box continuously at that resolution. Visual inspection shows the highlighted predators held against map boundaries or rock geometry for long periods.

The full-game example is seed 19099, ranked first by the sum of qualifying predator-seconds: 21 episodes totaling 27,506 seconds. Its highlighted predator remained within a 2.54×34.59-unit box from approximately t=275 to the game's end at t=2,773.2.

Media is intentionally kept outside Git. The persistent local gallery is served at http://localhost:9101/ and the full frame-by-frame seed-19099 replay at http://localhost:9100/.

Raw per-game findings are in `games.jsonl`; `selection.json` records the ten displayed intervals and their exact-frame verification.
