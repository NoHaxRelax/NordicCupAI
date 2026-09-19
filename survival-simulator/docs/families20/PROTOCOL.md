# Second ten-family campaign

Completed: all 40,800 games verified. See RESULTS.md for the ranking and training
trajectories, and COMPUTE.md for runtime, cost estimate and reproducibility checks.

Follow-up to families10. Ten **new parameter families / policy combinations**,
not ten newly trained neural networks. Native simulator and Orchard policy are
unchanged. The previous population winner provides the common starting policy.
The harvest and exploration winners seed relevant combinations. None of these
combinations is presumed better before fresh evaluation.

1. Expanded population: extend the previous optimum's boundary, including slack.
2. Population + harvest: jointly tune carrying capacity and food economics.
3. Population + exploration: jointly tune capacity and exploring versus settling.
4. Clustered orchards: favor nearby tree clusters and multiple residents.
5. Distributed orchards: spread residents and extend lone-agent reach.
6. Ripeness reserve: trade fruit growth against emergency hunger.
7. Mobile harvest: adjust relocation frequency and travel penalties.
8. Scheduled breeding: vary reproductive reserves across game time.
9. Trait succession: select replacement traits and prioritize nursery food.
10. Economy + gaze: combine population/food economy with watched-predator escape.

Per family: **60 evaluations** (one seeded candidate, nine random initial trials,
50 Gaussian-process expected-improvement trials), each on the same **64 new maps**,
seeds 8001–8064. Maximize mean final game score. Normal energy/aging/predators,
3000-second horizon or extinction, fixed policy seed 0 independent of world seed.
The native public-observation ABI remains unchanged.

Select one configuration per family strictly by training mean; freeze it before
the **200-map test**, seeds 9001–9200. Also evaluate the previous population winner
and original baseline on those same 200 maps. Never use test results in the search.
Total: 38,400 training games + 2,000 winner tests + 400 controls = **40,800 games**.

Use the ten existing Oscar CPU pods, one per family, 32 worker processes each.
They cost $0.96/hour each; leave them running afterward as requested. The user
removed the historical $10 cap. Each job still has a 90-minute fail-safe deadline,
not a minimum rental duration. Expect roughly 30–45 minutes from prior throughput;
report actual times and active compute cost at completion, separately from idle
pod uptime. No official competition validation attempts.

Runner: `scripts/tune_families20.py --family NAME --out results/NAME`.
Reporter: `scripts/report_families10.py docs/families20 --campaign 2`.
The manifest saves source/config hashes and environment details; per-game JSONL
and each candidate configuration are retained. Plot all 60 training steps and
compare test means and paired differences against the previous winner.
