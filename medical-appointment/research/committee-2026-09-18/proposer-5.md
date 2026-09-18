# Proposer 5: the widest lens (metric geometry, priors, generator, rules)

2026-09-18, Friday afternoon. Scope: everything outside the ASR/prompt/unit work the other proposers cover. Every number below was measured this afternoon on the 39 training conversations (turbo transcripts, turbo offsets, spans-on-no policy, the scorer from `utils.py`/`local_evaluator.py`), with scratch code under the session scratchpad (`proposer5/`: `labels_only.py`, `mbr_runs.py`, `anchor_ablation.py`, `neighbour_edges.py`, `energy_edges.py`, `sc_bench.py`, `sc_analyse.py`). Nothing under `bench/mine/` was read. No repo file was changed. Baseline for all comparisons: Qwen3.8-27B, `units-fewshot`, clause-and units, 0.8125 (H100 run) / 0.8179 (the same config re-run on the A100 pod; 191 of 195 spans identical, so same-config noise is about 0.005).

The short version: from this lens almost every door is closed, and it is closed by measurement, not opinion. The one idea that survives is worth about +0.003. The value of this report is that nobody needs to spend Friday night on the eleven things below that looked promising.

## Ranked list (expected gain x feasibility before Saturday morning)

### 1. Conditional start offset by boundary type (+0.002 to +0.003, 20 minutes, near-zero risk) — partly tested

Mechanism. The served start rule is one constant (first word's end - 0.20 s). On the oracle-selected clause-and run the residual (gold minus prediction) splits by whether the unit starts after a pause or mid-speech (a clause piece cut inside a sentence):

| start boundary | n | median residual | IQR | MAE |
|---|---:|---:|---:|---:|
| after a pause >= 0.3 s | 138 | -0.02 | -0.10..+0.04 | 0.118 |
| mid-speech (< 0.3 s) | 54 | +0.05 | -0.04..+0.12 | 0.206 |

End residuals show no such split (sentence end +0.02, comma +0.01, no pause after -0.02; MAE 0.04 to 0.08). So mid-speech starts should move 0.05 s later: `START_OFFSET` -0.22 after a pause, -0.15 mid-speech (`span_from_ids`, one `if` on the gap before the unit's first word).

Why it moves the score. 54 of 195 golds start mid-speech under clause-and units; 0.05 s on a 2.9 s span is 0.017 IoU each, 54/195 of that is 0.005 tIoU, 0.003 score, and only on spans where the model cites the right piece, so call it +0.002.

Test. Replay `qwen3.8-27b.units-fewshot.large-v3-turbo.clause-and*.json` with the two constants fitted leave-one-conversation-out (`bench/asr/fit_edges.py` style, 15 minutes on the laptop, no GPU). Untested part: the LOCO fit; the medians above are in-sample on the oracle run. Risk: none beyond the size of the gain.

### 2. Thinking mode on the 27B (0 ± 0.01, 20 pod-minutes) — untested

Mechanism. The pod's vLLM runs `--reasoning-parser qwen3`, so a schema-constrained answer after a `<think>` block works. `bench/llm/bench.py --no-think none --max-tokens 3000 --asr large-v3-turbo --variant units-fewshot` with `UNIT_SPLIT=clause-and`, `--out` to scratch. Prior: comprehension is not the residual (entries 36, 37, 41: frontier models plateau at the same spans), so the expectation is zero; it is on the list because it is the one unexplored axis that costs nothing but pod time. Latency risk if adopted: about 800 tokens per question, 10 in parallel, roughly 10 to 15 s per conversation on the A100, inside `LLM_DEADLINE=40` but with less margin for the fallback. Do not adopt without a paired replay over two standard errors.

### 3. Speaker-labelled transcript (0 to +0.005, 1 to 2 hours) — untested, low

Mechanism. Render units as `[12] D: ...` / `[13] P: ...` using MFCC two-cluster diarization (`bench/ref/tag_speakers.py` already does it for the reference files; the audio is two clean synthetic voices). What the labels measure on training: golds are doctor-only in 126 of 195, patient-only in 48, mixed in 21. The question's wording does not predict the speaker: 74 questions say "the patient"; their golds are DD 33, PP 29, PD 9, DP 3. So there is no prior to hand the model, only turn structure it mostly has already from the units. Expected gain is a rounding error; risk is a new serve-time component the night before the attempt. Listed for completeness, not recommended.

### 4. Serve-time insurance: exact match against training (0 expected, 0 risk, 10 lines)

If an evaluation conversation's transcript matches a training transcript above 0.95 word-sequence ratio and the question text matches, return the training gold. The README says the evaluation set is different, and the 19 validation conversations' nearest training neighbour is at ratio 0.17, so this pays nothing unless the organisers reuse a file. Zero cost; someone can add it while doing item 1.

## Closed doors (measured; do not spend time here)

Ordered by how tempting they looked.

**Cross-prompt ensembles (minimum-Bayes-risk under IoU).** Candidate spans from several stored 27B runs, the span chosen to maximise the summed IoU to all candidates over the product grid of their starts and ends, yes/no by majority. Same-config pair (H100+A100) 0.8153; three clause-and few-shot runs 0.8155; adding joint-demo, the `-cl` prompt and clause-all runs 0.8133 to 0.8159; the twelve Qwen runs 0.8085; adding the Opus and Sonnet probes 0.8156. Best single run 0.8179. The oracle best-of over the same candidates is 0.839 to 0.866, so the runs do disagree usefully, but consensus lands on the mean of the singles every time. Script `mbr_runs.py`.

**Self-consistency (n=8 samples at T=0.7, served prompt, pod run `sc_bench.py`).** Scored on the 160 questions finished when the budget ran out: greedy 0.806 (A100) / 0.799 (H100); MBR over 8 samples 0.800; samples plus greedy weighted 0.799 to 0.803; modal span 0.801; oracle over samples 0.849. Sample agreement is calibrated (unanimous spans: mean tIoU 0.80, n=41; 50 % agreement: 0.55, n=6) but 18 positives have a sample better than greedy by 0.1 and MBR picks it once. Shortest sampled span 0.777, longest 0.809: when samples disagree the longer one is right more often, a +0.003 hint inside noise at n=160, not a recommendation.

**Yes/no threshold.** With spans returned on every question (entry 38) the boolean only touches the 0.4 accuracy term, so the right threshold is 0.5 for a calibrated model. The 27B's P(yes) is binary on 378 of 390 questions; all four binary errors are missed positives (p 0.20, 0.10, 0.44, 0.04) whose spans are returned anyway at mean tIoU 0.62. LOCO-chosen threshold: accuracy 0.9846 against 0.9897 as answered. The whole accuracy loss is 4 x 0.4/390 = 0.004.

**Hedging between candidate spans.** For nested candidates U and U+N with belief p, E[IoU] as a function of how much of N is included is convex (second derivative 2 p u / (u+x)^3 > 0), so the optimum is an endpoint: commit. For disjoint candidates the union scores 0.5 (a+b)/(a+b+gap) < 0.5 at p = 0.5. There is no interval that beats committing to the likelier candidate, at any belief. The self-consistency policies above confirm it empirically.

**Neighbour inclusion is not a coin flip, and the default is already right.** On clause-and units a short reply (<= 4 words) directly after the fact unit is inside the gold 41 times and outside 99 times; a short question directly before it is inside 0 times and outside 14. "Exclude" is the right commitment on training; there is no feature to flip it with.

**Anchor rule (`model.anchor_ids`, quote unit plus cited ids within 2).** Replay with five alternatives (cited ids only, quote only, cited run containing the quote without the +-2 limit, strict adjacency): Qwen3.8 clause-and A100 0.8179 served / 0.8203 cited / 0.8159 run+quote; Qwen3.6 0.8139 / 0.8052 / 0.8103; joint-demo 0.8126 / 0.8138 / 0.8163 (strict adjacency); sentence-unit runs within 0.005. Signs disagree across models, and only 7 of 195 citations are more than two units from the quote. Neutral; leave it. Script `anchor_ablation.py`.

**Edge refinement from the audio (VAD/energy onset, the stable-ts hypothesis).** The audio's silence floor is -85 dBFS, so speech onsets are found trivially, and the annotated start still scatters around them: energy onset + LOCO constant, best of nine thresholds, start MAE 0.19 to 0.27 s against 0.148 for the served first-word-end rule; ends 0.10 against 0.077. Oracle-run tIoU drops from 0.919 to 0.86 to 0.89. The start jitter is on the annotation side (median residual 0.000, IQR +-0.08 s under the served rule), not in the timestamps, so no aligner will close it. Together with entries 18, 19, 29 and 42 the edge lever is closed from every side.

**Lexical alignment from the question (is the generator templated on the gold text?).** BM25-lite over units: the gold unit is the top hit for 127 of 195 questions, top-2 for 156; the top unit's mean tIoU is 0.524 against 0.70 for the 27B. Only 18 questions carry a number, the gold unit holds it in 14. Questions are paraphrases, not templates; lexical retrieval is a weaker selector and adds nothing as a tie-break.

**Question order and ids.** The ten questions arrive shuffled (37 distinct yes/no patterns in 39 conversations, yes count by position 12 to 25 of 39); the `qNN` in the id tracks gold time order in 1 of 39 conversations, and the id is not sent anyway.

**Position prior.** Gold midpoints by decile of the conversation: 9, 21, 28, 23, 29, 21, 29, 23, 9, 3. Flat over the middle; 4 golds in the first 5 s, none in the last 5 s. Useless for selection.

**Duplicates.** Most similar training pair 0.15 word-sequence ratio; nearest training conversation to any validation file 0.17. Distinct scripts throughout.

**Validation style versus training (no labels used).** Validation audio is longer (median 118 s vs 106 s, 321 vs 290 words, unit length 1.56 vs 1.41 s); question style is the same (tag questions 8 % vs 5 %, digits 8 % vs 7 %, median 8 words, same openers). Nothing to adapt to; longer conversations make selection slightly harder, which is a reason to expect validation a little under training, not a lever.

**Speaker prior.** See item 3: none from the question wording.

## What this lens says about where the remaining 0.19 of tIoU is

Every re-selection strategy (ensemble, sampling, retrieval, priors, thresholds) lands on the same spans because the disagreement between candidates is exactly the annotators' unresolved choices (entries 43, 45): the two-statement cases (about 17, 0.05 score, floor) and the neighbour calls (30/70, already committed the right way). The recoverable part on training is extent inside the cited unit, which the clause units are already taking. From here the honest expectation for Saturday is the served 0.81 on training minus the validation-length effect, and the work that matters tonight is robustness of the serving path, not the score.
