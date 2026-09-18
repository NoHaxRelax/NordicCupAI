# Overnight plan, 19 Sept 2026 (Claude, orchard session) — keep this short and current

## Oscar's instructions (23:55 and 00:20, paraphrased; also in memory overnight-directives-2026-09-19)
- First ~2 h (to ~02:00): make the population live as long as possible (no predators), everything in C++,
  run on Runpod only (never simulate on the laptop). Be creative; think at several abstraction layers.
- NEVER submit anything through the competition API.
- Then (and while waiting): study Lucas's latest bait/trap work, rewrite it in C++, and combine it with the
  survival policy: maximise lifetime while baiting/trapping predators as well as possible.
  Policy ideas: non-contributing agents run the survival policy; keepers stay near the trap to replenish
  bait; old agents (dying anyway) become guides/bait; agents keep enough energy to spawn a child when a
  predator is near; predators chase the closest agent; escapes by sprint, circling behind (turn-rate
  exploit), walking backwards, hiding behind walls; losing agents to predators should be rare; many
  predators late in the game is the most important regime; test the hypothesis that late game lacks the
  resources to keep loading the trap. When something fails, build a narrow test and fix it step by step.
- Keep working all night (cron tick every 10 min). Budget $100 new spend, up to 8 pods, until spent.
  Shut pods down when idle; boot more when needed within budget.
- Version control locally; push only to survival-simulator/oscar-* branches. Few replays; compact data;
  always save the best policy so we can return to it. Keep a structured plan; avoid rabbit holes.

## Layers and experiments
P1 no-predator survival (C++ fork survival/nightsim, runner nightsim/run.py, driver night.sh)
- L1 measure: 1000-seed baseline best vs r21s0c2 (base1k); harvest per 250 s (fruit spawned vs eaten),
  death classes by trees left. Question: late game supply-limited or discovery-limited?
- L2 parameters: paired search around the leader, 200-500 seeds per candidate, confirm on 1000+.
- L3 mechanisms in C++ (each behind a parameter, default off, A/B on 500+ seeds): chosen from L1 data.
- L4 population strategy (colony count/spread, late-game size).
P2 predators (after ~02:00): report lucas-trap-report.md (branch survival-simulator/lucas-experimental).
- T0 port harness: predators on in nightsim runner; measure current survival policy with predators.
- T1 escape tests (single predator, scripted): sprint, circle-behind, backwards, wall hiding -> loss rate.
- T2 guide-to-trap and bait-holding tests; T3 bait replacement by old agents; T4 full games; T5 many
  predators late; trap on/off comparison on the same seeds.

## Rules of thumb
- Judge by paired means on >=500 seeds (per-run sd ~330 s). Report survival relative to trees left.
- Every accepted change: BEST.md + git commit in teamrepo branch survival-simulator/oscar-overnight-cpp.
