# Committee report 03: where the next 0.05 to 0.10 of tIoU comes from

Reviewer lens: spans only. Written 2026-09-17 late, for Elias on the morning of the 18th.
Everything below is measured on the 39 training conversations (195 annotated positives) from the
stored result files under `bench/results/llm/` and the turbo transcripts. No benchmark was run, no
server contacted. The scripts are throwaway; the three that matter are reproduced as items 1 to 3
of the plan.

## Verdict

Two things are true and they point in the same direction. First, the cluster bench has been scoring
turbo transcripts with large-v3's edge offsets, because `model.py` reads the offsets from
`ASR_MODEL` at import while `bench.py --asr` only chooses which transcripts to read, and no job
script exports `ASR_MODEL`. Re-scoring all fourteen turbo runs from their own stored output with
turbo's fitted offsets adds 0.017 to 0.023 mean tIoU uniformly, so the honest training score of the
27B is 0.797, not 0.780, and Opus is 0.807, not 0.793. That is a measurement correction rather than
a production gain (the served pipeline sets `ASR_MODEL` and is already correct), but it resets every
variant comparison and it tells you that you are at the top of the leaderboard cluster, not below
it. Second, the remaining loss is not edges and not model capability: it is that a Whisper sentence
is the wrong unit for a third of the annotations. Of the 34 positives that all six probed models
score below 0.5, fifteen have a gold span covering one clause of a compound sentence, where no merge
of whole units can reach 0.5 but an oracle word range inside the very unit the model already chose
reaches 0.814. Clause-level trimming is the one lever I could confirm pays: splitting the chosen
unit at a comma followed by a coordinating conjunction and keeping the piece that matches the
question adds 0.012 to 0.016 mean tIoU on every one of the five runs I replayed, with no new
inference, and an oracle clause chooser would add 0.036. Everything else on the candidate list is
either measured dead (consensus choosers, pause thresholds, per-type offsets, the gap rule, the
quote tie-break, every hand-built neighbour rule) or needs a fitted re-ranker you cannot validate in
a day. Spend tomorrow morning on the offset fix, a replay harness, clause trimming, and one GPU
session that tests clause units in the prompt plus a one-line prompt change for the systematic
too-early bias. Expected honest gain 0.02 to 0.05 tIoU, that is 0.012 to 0.030 of score.

## 1. The 34 shared hard positives

Cut as: mean tIoU below 0.5 for all six of Qwen3.8-27B (units-fewshot and units-joint-demo),
Qwen3.6-27B (units-fewshot), Opus 5, Sonnet 5 and Haiku 4.5 (all units-joint-demo), with turbo's
correct offsets applied. With the offsets the runs were actually scored with, the count is 43, which
is the "roughly 50" in the brief; the offset fix alone rescues nine of them. Entry 36 counted about
70 per model and 52 shared by three; the tighter six-model intersection is 34.

Categories are assigned in priority order, so each question appears once.

| category | n | tIoU now | oracle unit merge | oracle clause piece | oracle word range | oracle neighbour | median gold length |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Sub-sentence gold, no unit merge reaches 0.5 | 15 | 0.343 | 0.372 | 0.682 | 0.814 | 0.343 | 1.24 s |
| B. Wrong sentence, prediction disjoint from gold | 9 | 0.000 | 0.886 | 0.000 | 0.000 | 0.019 | 2.94 s |
| C. Gold is a question plus its answer | 0 | | | | | | |
| D. Gold spans 2 or more units, other kinds | 8 | 0.352 | 0.894 | 0.352 | 0.352 | 0.627 | 7.57 s |
| E. Sub-unit gold inside one unit, no clause marker | 2 | 0.417 | 0.796 | 0.417 | 0.936 | 0.796 | 1.77 s |
| the other 161 positives, for contrast | 161 | 0.877 | 0.899 | 0.890 | 0.930 | 0.896 | 2.92 s |

The oracle columns are computed inside the unit run the best model already chose, so they are what a
perfect post-processor would reach without changing the selection. Non-exclusive tags on the same 34:

| property | count |
|---|---:|
| the chosen unit contains a clause boundary | 15 |
| gold end falls mid-utterance | 12 |
| gold covers 2 or more units | 12 |
| prediction disjoint from gold | 11 |
| gold start falls mid-utterance | 9 |
| gold shorter than 1.3 s | 9 |
| gold covers 3 or more units | 7 |
| first gold unit ends in a question mark | 2 |

Reading the classes.

Class A is one pattern repeated: the doctor packs two facts into one sentence and the annotators mark
one of them. Sample 69 has four questions on "Your blood pressure is normal, and your foot status is
normal." and "Overall, your diabetes is stable, with no signs of complications."; sample 67 has three
on "Taken together, your diabetes is stable, with good self-care on your part." and "We continue your
treatment unchanged, and keep to the regular checkups."; the same shape appears in samples 20, 54, 71
and 90. Thirteen of the fifteen are reachable by cutting at the comma. The two that are not are
annotation errors: sample 63 question 2 ("The lipid profile came back normal, didn't it?") has a gold
of 0.00 to 0.26 s sitting on "Good morning, Anne." while the lipid profile is actually stated at
47.44 to 52.86, and sample 64 question 2 has a 0.16 s gold on "Good afternoon.". Those two cap the
achievable mean tIoU at about 0.99 and cost 0.010. Nobody should chase them.

Class B is not a granularity problem, it is a which-mention problem, and it is directional. In 7 of
the 9 the model picked an earlier mention than the annotators did, and across the whole training set
the same bias holds: for Qwen3.8-27B units-fewshot, 40 of the 60 positives below 0.5 have the
model's span earlier than the gold and only 20 later; Qwen3.6 gives 40 against 17 and Opus 35
against 24. The pattern in the text is consistent: the model marks where the topic is first raised
(often by the patient) and the annotators mark the later, more explicit statement, usually the
doctor naming the finding or stating the plan. Sample 37 wants "A tick bite." and gets "I found a
tick attached to my leg."; sample 20 wants the prescription being created at 182 s and gets the
patient's request at 42 s; sample 17 wants "After a meal every day for 2 weeks." and gets "100 mg
daily for 2 weeks.". Two of the nine (samples 57 and 70) look to me like the annotators being loose
rather than the model being wrong. No post-processing can reach this class: clause and word-range
oracles are 0.000 on it and the neighbour oracle is 0.019, against an oracle unit merge of 0.886.
Only the prompt can.

Class D is long golds, median 7.57 s, where the annotators marked a whole exchange of three to six
units and the model cited one or two. A plus-or-minus-one neighbour choice takes this class from
0.352 to 0.627, the only class where the neighbour lever does real work, but it needs 3 to 6 units
on some of them, not one.

Class C is empty in the hard set, which is worth writing down: the question-plus-answer convention
noted in entry 28 does exist (eight cases among the other 161 positives) but the models already
handle it. It is not where the loss is.

## 2. The levers, ranked by gain per hour

Base for every number: Qwen3.8-27B units-fewshot, turbo transcripts, turbo offsets, mean tIoU 0.6692
over the 195 annotated positives, accuracy 0.990, score 0.7974. Score gain is 0.6 times the tIoU
gain. The noise floor is 0.006 on the training bench and about 0.010 on a validation run.

| # | lever | oracle headroom | realistic gain | score gain | cost | status |
|---|---|---:|---:|---:|---|---|
| 1 | Export `ASR_MODEL` so the bench uses the tag's own offsets | n/a | +0.021 measured | +0.013 on the bench table only | 10 min, no GPU | measured on all 14 runs |
| 2 | Sub-unit trim: clause piece of the chosen unit, chosen by question overlap | +0.036 | +0.012 to +0.016 measured | +0.007 to +0.010 | 1 h, no GPU | measured on 5 runs |
| 3 | Clause units in `make_units`, model cites the clause | merge ceiling 0.857 to 0.893, single 0.728 to 0.741 | +0.02 to +0.04 | +0.012 to +0.024 | 1 to 2 h plus one 2-min GPU bench | ceiling measured, realised untested |
| 4 | Prompt line: prefer the later, more explicit statement | up to +0.041 (the 9 class-B at 0.886) | +0.01 to +0.02 | +0.006 to +0.012 | 20 min plus one GPU bench | bias measured, fix untested |
| 5 | Sub-unit trim at word level rather than clause level | +0.129 | unknown, needs a word-range selector | up to +0.077 | half a day plus GPU | oracle only |
| 6 | Smarter neighbour rule, fitted re-ranker | +0.081 | +0.02 to +0.03 at best | +0.012 to +0.018 | 3 to 4 h, high overfit risk | three hand rules measured, all negative |
| 7 | Consensus over prompts or models with a chooser | +0.056 (oracle best-of-5) | +0.005 | +0.003 | 2 to 5x inference | measured dead |
| 8 | Per-question-type edge offsets | ~0 | 0 | 0 | 1 h | measured dead |
| 9 | Different pause threshold | 0.000 | 0 | 0 | 20 min | measured dead |

Detail on each.

**(a) Sub-unit trimming.** Two versions. The clause version splits the chosen unit run at a comma,
semicolon or colon followed by `and`, `but`, `with`, `so`, `then` or `or`, then keeps the piece whose
content words overlap the question most and builds the span from that word range with the same
first-word-end and end offsets. 33 of the 195 chosen runs contain such a cut. Replayed over the
stored output with no new inference: Qwen3.8-27B units-fewshot 0.6692 to 0.6812, Qwen3.6-27B
units-fewshot 0.6702 to 0.6833, Qwen3.8-27B units-joint-demo 0.6594 to 0.6762, Opus 5 0.6775 to
0.6932, qwen3:4b units-joint-demo 0.5661 to 0.5782. Positive on all five, between 0.012 and 0.016,
so it is above the training noise floor and it is the only lever I can hand you already measured.
An oracle clause chooser on the same candidate set reaches 0.7048, so the simple question-overlap
argmax captures about a third of what is there and a model-side choice could take the rest. Do not
trim to the model's verbatim quote instead: it helps Qwen3.6 by 0.013 but costs Opus 0.072, because
Opus quotes tightly and the difflib window match mislocates it. The word-level oracle inside the
chosen run is 0.798, which is where the remaining 0.129 sits, but nothing in the repo selects a word
range that well and the `words` variant already lost 0.05 against `units` on every model (entry 34).

**(b) Smarter neighbour rule.** The oracle over six options (as-is, plus previous, plus next, plus
both, drop first, drop last) is 0.7504 against 0.6692, so +0.081 is genuinely there. It is also
genuinely hard to get: as-is already wins on 146 of 195, and the 49 that want something else split
almost evenly between plus-previous (13), plus-next (13), drop-first (14) and drop-last (8), so the
rule has to be right per case. Always-plus-previous scores 0.469 and always-plus-next 0.445. I tested
three principled conditioned rules, all motivated by the annotation conventions in entry 28, and all
three lose: "the previous unit is a question and the gap is under 0.9 s, so include it" fires 49 to
55 times and costs 0.060 to 0.070 on every run; "the span is under 1.6 s and the next unit starts
within 0.35 s, so add it" costs 0.006 to 0.013; "the chosen unit ends in a question mark, so add the
answer" fires 0 to 5 times and does nothing, because the models almost never cite a question unit.
That three plausible rules all lose is the reason I rank a fitted re-ranker below clause units: the
signal is not in the features a rule can see, and 195 spans with 49 informative decisions will not
support a trustworthy LOCO fit by tomorrow evening.

**(c) Consensus.** The oracle best-of over five servable runs is 0.7255 and over all eight runs
including the three Claude models 0.7665, which restates entry 36's 0.719. Every chooser I could
build lands inside noise of the best single run (0.6702): medoid 0.6705, majority over cited unit
ids 0.6739, union 0.6511, intersection 0.6339. Narrower pools do no better (three prompts on one
27B: best single 0.6692, medoid 0.6675, majority 0.6676). The models agree on the easy spans and
disagree unhelpfully on the hard ones, and no cheap statistic tells you which disagreement to
believe. Against 2 to 5 times the inference and the same multiple of serving risk, this is not worth
starting.

**(d) Unit segmentation.** The pause threshold is inert. `PAUSE_SPLIT` at 0.45, 0.60, 0.80, 1.20 and
effectively infinity all give exactly the same ceilings (single unit 0.728, merge of up to four
0.857, 59 units per conversation); only 0.35 differs and it is worse (0.715). Only 101 of the 2305
training units end without terminal punctuation, so terminal punctuation and Whisper's own segment
boundaries do nearly all the cutting and the pause rule is almost redundant. Do not sweep it.
Punctuation is the other story: adding the comma-plus-conjunction cut raises the single-unit ceiling
from 0.728 to 0.741, the merge-of-four ceiling from 0.857 to 0.893, and drops the positives whose
best single unit is under 0.5 from 43 to 36, at 62 units per conversation instead of 59. That is
idea 7 of `06-edge-ideas.md` confirmed on turbo (the file's 0.882 was measured on large-v3). Doing
it in `make_units` rather than as a post-hoc trim is better because the model then cites the clause
itself, and the few-shot examples already show it the annotators' exact marked stretch, so it has
what it needs to match the granularity. Render as `[12a]` and `[12b]` under the sentence number as
the idea file says, so the visible id space does not double.

**(e) Per-question-type edge offsets.** No signal. Grouped by question first word (`was`, `is`,
`does`, `did`, `has`, `the`, `were`, `will`, `are`), by tag question against plain, by one, two or
three cited units, and by clause-joined against simple units, every group with n at least 8 has the
same residual median within 0.06 s at the start and 0.06 s at the end. The apparent residual in the
stored runs (start -0.04, end -0.14) is the `ASR_MODEL` bug and nothing else. A grid search on the 0.02 s annotation grid over
the realised spans of the best run puts the optimum at start -0.22 and end 0.00, worth 0.6700 against
0.6692 for turbo's LOCO-fitted -0.20 and -0.02, so re-fitting the pair on the realised spans buys
0.0008 and is not worth a commit. After lever 1 the edges are unbiased in every grouping. Entry 5's
conclusion still holds: the remaining edge error is inside Whisper's own boundaries and no formula
gets at it.

**(f) From the ideas file, supported by this analysis.** The clause units of idea 7 are lever 3 and
should be built. The "later, more explicit statement" prompt line is new and follows from the
directional bias in section 1: 40 of 60 sub-0.5 cases too early against 20 too late, consistently
across three model families. One sentence in `SYSTEM`, one GPU bench to check it does not disturb the
easy cases. Two items from idea 10 are measured dead: tightening `span_from_ids`'s gap rule from "at
most two" to strict adjacency changes mean tIoU by -0.001 to +0.002, and flipping `locate_quote`'s
tie-break from the earliest best match to the latest changes nothing at all, because difflib ratios
never actually tie. Idea 8's yes-threshold is irrelevant now: the 27B answers 192 of 195 positives
yes and has no false yes on the 142 hard negatives, so there is no probability mass to move.

## 3. The plan for tomorrow

In this order. Items 1 to 3 need no GPU and no network and should be finished before the pod is
touched; items 4 and 5 share one pod session.

1. **Make the bench use the right offsets (10 minutes).** In `bench/hpc/llm_bench.lsf` and
   `/workspace/run_benches.sh`, export `ASR_MODEL=$ASR` inside the ASR-tag loop. In
   `bench/llm/bench.py`, assert at start-up that `model.ASR_MODEL == args.asr` and refuse to run
   otherwise, so the two can never diverge again. Separately, confirm the serving environment
   exports `ASR_MODEL=large-v3-turbo`: `model.py` defaults to `large-v3`, and that default would
   both load the wrong weights and apply the wrong offsets with no error message. Read: the
   `offsets` line `bench.py` already prints at start-up must say start -0.20 end -0.02.

2. **Write `bench/llm/replay.py` (30 minutes).** It takes a stored result JSON, re-runs only the
   post-processing over `questions[].raw` against the transcripts named in its config, and reprints
   the summary block. Validate it by reproducing each file's stored `mean_tiou` exactly with
   `START_OFFSET=-0.14 END_OFFSET=0.12`, then report the turbo-offset number. This is the enabler
   for everything else: with it, every post-processing idea becomes a five-second test across all
   fourteen stored runs instead of a GPU job. Expect the fourteen runs to move up by 0.017 to 0.023,
   with Qwen3.8-27B units-fewshot at 0.6692 and score 0.797.

3. **Clause trimming in the replay, then in `model.py` (1 hour).** Cut the chosen unit run at a
   comma, semicolon or colon whose next word is `and`, `but`, `with`, `so`, `then` or `or`; score
   each contiguous merge of the resulting pieces by content-word overlap with the question, ignoring
   the stop list already in `prompts.py`; build the span from the winning word range with the
   existing offsets. Gate it behind `CLAUSE_TRIM=1`. Read: mean tIoU on all five of Qwen3.8-27B
   fewshot, Qwen3.8-27B joint-demo, Qwen3.6-27B fewshot, Opus 5 and qwen3:4b joint-demo. Ship only
   if every one improves; the numbers to beat are 0.6692, 0.6594, 0.6702, 0.6775, 0.5661, and I
   measured 0.6812, 0.6762, 0.6833, 0.6932, 0.5782.

4. **Clause units, with the pod up (1 to 2 hours, about 12 minutes of it on the GPU).** Add the same
   cut to `model.make_units` behind `CLAUSE_UNITS=1` and render the clause ids as `[12a]`, `[12b]`.
   First confirm the ceiling offline with `span_ceiling.py` and `bench/asr/compare.py`: single unit
   0.728 to 0.741, merge of up to four 0.857 to 0.893. Then restart the RunPod pod per
   `bench/hpc/RUNPOD.md` (8 minutes to a serving vLLM) and run one `units-fewshot` bench on
   Qwen3.8-27B, which is about 2 minutes of wall time for all 390 questions. Read: mean tIoU against
   0.6692 and the count below 0.5 against 60. Take it if it clears 0.69; this is the only item with a
   plausible path past 0.70.

5. **The later-mention prompt line, same pod session (20 minutes plus 2 minutes of GPU).** Add to
   `model.SYSTEM`, next to the near-miss paragraph: when the transcript establishes the same fact
   more than once, quote the later and more explicit statement, typically the doctor naming the
   finding or stating the plan, rather than the first time the topic comes up. Rerun the same bench.
   Read three numbers, not one: mean tIoU, the count of disjoint predictions that sit earlier than
   the gold (11 now for this run), and accuracy, because the line touches every question and the
   accuracy at 0.990 is the thing you cannot afford to lose. Keep it only if tIoU gains at least
   0.010 and accuracy does not drop.

6. **Then stop.** Whatever survives items 1 to 5, freeze it, and spend the remaining time on one
   validation run of the 27B configuration end to end. The 27B pipeline has never been scored by the
   portal; every number in this report is training-set, and entry 24 is the standing warning that a
   local gain need not transfer. A validation run cannot resolve anything under 0.010, so do not
   spend runs A/B-testing the levers; spend one confirming the whole thing works and holds its
   score.

## 4. What not to attempt with the time left

- **Consensus, self-consistency or any multi-model vote.** Measured: every implementable chooser
  lands within 0.005 of the best single run while multiplying inference by two to five. The oracle
  is real (+0.056) and unreachable with anything cheap.
- **A pause-threshold sweep.** Measured inert from 0.45 to infinity. The idea that the unit
  boundaries depend on it is simply false on this data.
- **Per-question-type or per-anything edge offsets.** Measured: no group differs from any other by
  more than 0.06 s once the `ASR_MODEL` bug is fixed. Fitting them would be fitting noise on 195
  spans.
- **A fitted neighbour re-ranker.** The oracle is the largest single number in this report after
  word-level trimming, and I still say no: three principled hand rules all lose, two of them badly,
  which means the decision is not in the observable features, and a learned version on 49
  informative cases cannot be trusted on a day and a half. If you have spare hours after item 5,
  this is the one to spend them on, not before.
- **Word-level sub-unit selection.** The 0.129 oracle is the biggest prize on the board and there is
  no selector for it. The `words` variant is the existing attempt and it loses 0.05 to `units` on
  every model, and with few-shot examples it over-brackets 94 of 195 spans (entry 30). Clause units
  are the same idea with a candidate set small enough to choose from.
- **A bigger selector, or another Claude probe.** Entries 36 and 37 plus the six-model intersection
  here: Opus 5 is 0.008 tIoU above the 27B on the clean subset and fails on the same 34 spans.
  Capability is not the constraint and the probe is outside the rules anyway.
- **The nine class-B wrong-sentence cases as an engineering target.** No post-processing reaches them
  and hand-auditing nine questions will not generalise. The one honest shot at them is item 5's
  prompt line.
- **The two annotation errors** (sample 63 question 2, sample 64 question 2). Both golds are
  fractions of a second on the opening greeting with the real evidence elsewhere. They cost 0.010 of
  mean tIoU and no design recovers them.
- **Another ASR or aligner.** Entry 29 settled this across twelve configurations; turbo's end
  fingerprint is unique and nothing else is within 0.03 of its ceiling.
