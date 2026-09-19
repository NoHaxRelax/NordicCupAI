# Shared-food campaign conclusions

All 640 BO trials (128,000 checkpoint continuations), 4,800 independent validation continuations and 24,000 full games are complete. Frozen candidates were evaluated on 1,000 identical fresh map seeds each.

Expanded local food remains the best full-game model: mean 1676.4, 95% CI 1652.5–1701.0. Best new full-game mean is congestion_pricing at 1648.6, CI 1622.6–1675.2; paired difference −27.8, CI −54.2 to −0.7. Intervals are pointwise, without correction for model selection/multiple comparisons.

All 20 families improve fresh late-game continuations but lose score when deployed for the entire game. Training gains range +54.6 to +146.4; independent continuation gains +49.9 to +131.9. Train-minus-validation gaps range +0.9 to +43.8. This shows that checkpoint improvements survive new maps, but do not transfer to full-game use. It does not establish the cause: initialization, earlier behavior and resulting population/resource states change together. The validation-to-full difference is not an overfitting estimate. Do not replace the current best policy with these candidates.

Sharing alone gives +96.0 on fresh continuations but −38.9 on full games. This implementation also adds substantial compute: whole-loop CPU time rises from 0.889 ms/tick for expanded local food to 8.719 ms for the sharing control and 9.159 ms for congestion_pricing. These are population ticks under concurrent benchmark load, not individual agent actions or isolated production latency. Fewer predator deaths per game are not enough to infer better avoidance: population and lifetime exposure differ.

A useful next experiment would switch behavior only in demonstrably late-game states and train against full-game score, with an explicit compute constraint. That has not been tested by this report.

Nikolaj's recommended checked-in rank1 at e8c86288 scored 1474.4 (CI 1449.3–1499.0) on the same 1,000 seed numbers on the SSH PC. CPU, engine revision and Python/NumPy differ; this is a separate benchmark, not a controlled isolated policy comparison. It does not evaluate the unavailable untracked winner mentioned upstream.

Attributed active compute totals $37.88 across training, validation and full-game evaluation, excluding setup, idle time and storage; this is not an invoice total. CPU pods are left running as requested.

- [Every family's training, validation and full-game gains](COMPARISON.md)
- [Full-game scores, confidence intervals and timing](final/RESULTS.md)
- [BO evolution chart](training-evolution.png)
- [Nikolaj benchmark and provenance](../nikolaj-e8c86288/RESULTS.md)
