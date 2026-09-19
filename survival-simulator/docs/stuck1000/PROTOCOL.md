# Expanded local food: predator confinement census
Frozen configuration docs/localfood20/pod-7/expanded_local_food-winner.json, source parent 83fd242. Natural predators, policy seed 0, horizon3000s, seeds19001–20000. No policy changes and no spectator data passed to the policy.

Run on ssh pc, 12 workers, /home/lucas/expanded-stuck1000/survival-simulator. Python /home/lucas/entrapment-research-20260918/.venv/bin/python. Native engine and policy built there. census.pid/census.log; results/stuck1000 contains incremental games.jsonl and compressed per-game predator trajectories.

Screening definition: predator's sampled positions fit within a40×40 unit bounding box for strictly more than60 simulated seconds. Sample once per second. Predator identity is append-only native vector index (engine never removes predators). Store each trajectory for alternative spatial thresholds. Greedy longest nonoverlapping windows represent episodes, not every overlapping mathematical subinterval. This identifies confinement, not necessarily obstacle trapping; resting, orbiting, and bait pursuit are possible causes. Whole game ranked by sum of qualifying predator-seconds, count as tiebreak.

After all1000 runs: select ten distinct games, record every0.1s native tick using actual game draw methods and verify score/time against uninterrupted replay. Verify selected confinement interval at full frame resolution. Render highlighted90s-context clips at4× speed and one full-game video at20× speed. Also provide full interactive frame-by-frame replay of the game with greatest total confinement time. Keep exact-confirmation flags and do not call unconfirmed candidates continuous60-second traps.

Commands:
- census: scripts/find_stuck_predators.py --out results/stuck1000 --workers12 (use space before12)
- media: scripts/render_stuck_cases.py results/stuck1000
Progress: ssh pc 'tail -f /home/lucas/expanded-stuck1000/survival-simulator/census.log'

The existing Runpod campaign continues independently; its pods must not be stopped or overwritten.
