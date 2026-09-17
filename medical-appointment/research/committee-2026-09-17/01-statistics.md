# Committee report 01: statistics and measurement

Reviewer lens: statistics and measurement. Written 2026-09-17 late, for Elias tomorrow morning.
Everything below is computed from files already in the repository. No benchmark, no network call, no
portal request was made. The scripts that produced each number are described inline so you can redo
them; they are all short read-only passes over `bench/results/llm/*.json`, `data/question_train.csv`,
`bench/mine/span_state.json` and `request_dump/answers.jsonl`.

## Verdict

The arithmetic in the findings log is careful and the ASR work is the best-supported part of the day,
but three measurement problems make a large fraction of entries 30 to 37 non-comparable, and one of
them points the wrong way on a decision you have already acted on. First, the live scorer does credit
a span returned next to a no answer: the request dump of run F reconstructs to 0.67591704 under the
spans-on-no policy against the portal's reported 0.6759, and to 0.6590 under nulls-on-no, so entry 15
is refuted by a deterministic, exact reconstruction and every bench row you have been quoting under
nulls-on-no understates the live score by 0.005 to 0.010. Second, every bench run from 19:32 onward
was scored with large-v3's edge offsets (-0.14 / +0.12) applied to large-v3-turbo transcripts, because
`model.py` keys the fitted offsets off the `ASR_MODEL` environment variable and nothing sets it for
`bench/llm/bench.py --asr large-v3-turbo`; correcting that arithmetically adds 0.011 to 0.014 to every
turbo row and reverses the sign of entry 30. Third, the 0.006 "noise floor" is a same-code
reproducibility figure, not a standard error: the clustered standard error of the training score is
0.014 to 0.021, the paired standard error between two different systems on the same 39 conversations
is 0.008 to 0.011, and the projected standard error of a single 19-conversation run is 0.022 to 0.030,
so nearly every prompt and model ranking in the log sits inside one or two standard errors. Separately,
the recovered validation spans are 1.12 s longer on average than the training spans (95 percent
confidence interval +0.51 to +1.72 s), which means the annotation convention you fitted to is not the
convention you will be scored against, and the one prompt instruction written after the validation
labels were recovered was read off those labels.

## Findings

### 1. The live scorer does credit spans on no answers. Entry 15 is wrong.

**Claim in the log.** Entry 15: "the live scorer does not credit spans returned with a no answer ...
`local_evaluator.py` overstates our score by crediting those spans", with the consequence "read local
tIoU from the nulls-on-no policy in `bench/llm/bench.py`". Entry 28 repeats the confusion: "the portal
said 0.6759 for run F; the dump may hold the run before it".

**Evidence.** `request_dump/answers.jsonl` holds 19 conversations, 190 questions, with 80 of the 190
carrying a span next to a `false` answer, so it is a `SPAN_ON_NO=1` payload. Scoring it against the
hand binaries (`bench/mine/agent_answers.md`) and the recovered gold spans
(`bench/mine/span_state.json`) gives accuracy 181/190 = 0.95263158 and

| policy | reconstructed score |
|---|---|
| spans credited on no | 0.67591704 |
| nulls on no | 0.65902734 |

The portal reported 0.6759 for run F (entry 26). The spans-on-no reconstruction matches to four
decimals; the nulls-on-no reconstruction matches nothing. A four-decimal coincidence across 190
questions and 95 spans is not credible. Entry 27 already computed the decisive number,
`(0.6759 - 0.4 x 0.953) / 0.6 = 0.49`, and then attributed it to "the local bench", but 0.4914 is the
spans-on-no tIoU of that dump while the nulls-on-no tIoU is 0.4633. The evidence was sitting in the
log.

**Why entry 15's A/B test missed it.** The expected A minus B gap was assumed to be about 0.03, from
"share of positives answered no, about 0.15" times "their span tIoU, about 0.4". The true values from
the dump are 9 of 95 positives answered no (0.095), with span tIoU against the recovered gold of
0.0, 0.018, 0.0, 0.909, 0.0, 0.952, 0.0, 0.795, 0.0, mean 0.297. The expected gap is therefore
0.6 x (9/95) x 0.297 = 0.0169, not 0.03. Against a per-run standard deviation of about 0.007 (entry 26's
three runs), the difference of two independent runs has a standard deviation of about 0.010, so the
observed 0.0024 sits 1.5 standard deviations below the alternative hypothesis. That is not a rejection.
It is also possible that run B's `SPAN_ON_NO=0` never took effect, in which case A and B were the same
configuration and 0.0024 is pure noise, which is exactly what it looks like. Entry 15 does not record
run B's yes count, so this cannot be checked from the log.

**Severity: critical.** It changes the sign of a served-policy decision, the level of every quoted
bench number, and the value of the yes-threshold work (if a missed positive keeps its span tIoU, a
recovered positive is worth its accuracy point of 0.0021 plus only the *improvement* in tIoU, not the
whole of it).

**What to do.** Treat `local_evaluator.py` as correct as written and `stats.nulls_on_no` as a
diagnostic, not the headline. Rewrite entry 15 and add the reconstruction as the evidence. The
re-test costs zero portal runs: align `request_dump/answers.jsonl` with the run whose portal score
you know, reconstruct under both policies, and read off which one matches. The two policies differ by
0.0169 on that dump, which is far above the reconstruction's own precision (it is deterministic).

### 2. Every bench run after 19:32 used large-v3's edge offsets on turbo transcripts

**Claim in the log.** Entries 30, 32, 33, 34, 35, 36 and 37 all report numbers "on the turbo
transcripts", and entry 19 fixed the served configuration as "large-v3-turbo, START_RULE
first-word-end, START_OFFSET -0.20, END_OFFSET -0.02".

**Evidence.** The `config` block of every result file records the offsets in force.
`qwen3-4b.units.large-v3-turbo.json` (17:11), `units-v2` (17:20), `units-claim` (17:13),
`words` (17:16) and `units-pyes` (17:32) all carry `start_offset -0.2 / end_offset -0.02`. Every file
from `qwen3-4b.words-fewshot` (19:35) onward, including all cluster runs and all three Claude probe
scorings, carries `-0.14 / 0.12`. Those are large-v3's fitted values. The cause is in `model.py`
lines 66 and 81: `ASR_MODEL = os.environ.get('ASR_MODEL', 'large-v3')` and
`_fwe, _us, _end = _FITTED.get(ASR_MODEL, _FITTED['large-v3'])`. `bench/llm/bench.py` picks the
transcript with `--asr` and never touches `ASR_MODEL`, and `grep -rn ASR_MODEL` over the repository
finds nothing that sets it for the bench. So the offsets in a bench run depend on the shell that
launched it.

Re-scoring the stored spans arithmetically (shift each span by the difference of the two offset pairs,
re-clamp to the audio duration, recompute tIoU against the same gold) gives:

| run | as run (-0.14/+0.12) | at turbo's rule (-0.20/-0.02) | in-sample optimum |
|---|---|---|---|
| qwen3:4b units-fewshot | 0.6991 | 0.7096 | -0.22/+0.00, 0.7105 |
| qwen3:4b units-joint | 0.7102 | 0.7212 | -0.26/+0.00, 0.7223 |
| qwen3:4b units-joint-demo | 0.7241 | 0.7356 | -0.22/+0.00, 0.7366 |
| Qwen3.8-27B units | 0.7622 | 0.7732 | -0.22/+0.00, 0.7735 |
| Qwen3.8-27B units-fewshot | 0.7801 | 0.7927 | -0.22/+0.00, 0.7931 |
| Qwen3.8-27B units-joint-demo | 0.7748 | 0.7888 | -0.22/+0.00, 0.7895 |
| Claude Opus 5 units-joint-demo | 0.7927 | 0.8065 | -0.22/+0.00, 0.8069 |
| Qwen3.8-27B units, large-v3 transcripts | 0.7564 | 0.7409 | -0.14/+0.12, 0.7564 |

Two side results worth keeping. The large-v3 row is already at its in-sample optimum, which confirms
the diagnosis. And the in-sample optimum beats the LOCO rule by only 0.0004 to 0.0010 in score, so the
LOCO edge fit of entry 5 is essentially unbiased and the in-sample optimism you worried about in
entry 29 is worth about one thousandth, not a hundredth.

**Severity: critical.** It makes the entry 30, 32 and 33 comparisons apples to oranges (their
baseline, `units-pyes` at 17:32, was at -0.20/-0.02 while the variants were at -0.14/+0.12) and it
understates every turbo number in entries 34 to 37 by 0.011 to 0.014.

**What to do.** Make the offsets a function of the transcript tag, not of the environment. The
cleanest fix given the existing code is for `bench/llm/bench.py` to set `os.environ['ASR_MODEL'] = a.asr`
before importing `model`, and for `model.py` to raise rather than silently fall back when the tag is
not in `_FITTED` (research/06 already proposed exactly this as an `offsets.json` keyed by tag). Then
re-derive the headline table; do not re-run the models, the arithmetic shift above is exact apart from
the clamping edge cases.

### 3. Corrected headline table under the live policy and turbo's own offsets

Combining findings 1 and 2, with the clustered standard error from a bootstrap over the 39
conversations (3,000 resamples of whole conversations, which is the right resampling unit because the
ten questions of a conversation share one transcript):

| configuration | log | accuracy | tIoU | corrected score | SE |
|---|---|---|---|---|---|
| qwen3:4b units | 0.6996 | 0.9667 | 0.5319 | 0.7058 | 0.020 |
| qwen3:4b units-fewshot | 0.6991 | 0.9590 | 0.5539 | 0.7159 | 0.019 |
| qwen3:4b units-joint | 0.7102 | 0.9769 | 0.5592 | 0.7263 | 0.019 |
| qwen3:4b units-joint-demo | 0.7241 | 0.9897 | 0.5661 | 0.7356 | 0.021 |
| Qwen3.8-27B units | 0.7622 | 0.9872 | 0.6465 | 0.7828 | 0.016 |
| Qwen3.8-27B units-joint | 0.7613 | 0.9897 | 0.6484 | 0.7849 | 0.017 |
| Qwen3.8-27B units-joint-demo | 0.7748 | 0.9949 | 0.6594 | 0.7936 | 0.016 |
| Qwen3.8-27B units-fewshot | 0.7801 | 0.9897 | 0.6692 | 0.7974 | 0.016 |
| Qwen3.6-27B units-fewshot | 0.7739 | 0.9872 | 0.6702 | 0.7970 | 0.014 |
| Claude Haiku 4.5 joint-demo | 0.7391 | 0.9487 | 0.6226 | 0.7531 | 0.027 |
| Claude Sonnet 5 joint-demo | 0.7510 | 0.9462 | 0.6410 | 0.7631 | 0.033 |
| Claude Opus 5 joint-demo | 0.7927 | 1.0000 | 0.6775 | 0.8065 | 0.017 |

The "log" column is the number in the findings log. Note that Qwen3.6-27B and Qwen3.8-27B with
few-shot are separated by 0.0005 once both are corrected, which is a thousandth of a standard error.

**Severity: high**, because entry 34's decision text reads as if Qwen3.8 were the better model.

**What to do.** Replace the entry 34 and 35 tables with the corrected column and put the standard
error next to every score in the log from now on.

### 4. The noise floor is the wrong yardstick, and the right ones are two to three times larger

**Claim in the log.** Entry 15: "two identical runs differ by about 0.006 on ten conversations".
Entry 26: "validation-to-validation noise is about 0.01". These two numbers are then used throughout
entries 20, 22, 24, 30, 32, 33, 35 and 37 as the threshold for calling something a gain.

**Evidence.** `noise-run1.json` and `noise-run2.json` are two runs of the same code on the same ten
conversations. They differ on 8 spans of 100 and 2 answers of 100; the per-question tIoU difference
has standard deviation 0.070, so the paired standard error of the score difference is
0.6 x 0.070 / sqrt(50) = 0.0057. That reproduces the quoted 0.006, and it is the right number for
exactly one question: does re-running the same code give the same answer. It is not the right number
for anything else.

Three other quantities matter and none of them is 0.006:

| quantity | value | how it was computed |
|---|---|---|
| clustered SE of the score on the 39 training conversations | 0.014 to 0.021 | bootstrap over conversations, per-run, table in finding 3 |
| paired SE of the difference between two systems on the same 39 conversations | 0.008 to 0.011 | paired bootstrap over conversations |
| projected SE of a single run on a 19-conversation set | 0.022 to 0.030 | bootstrap of 19 conversations drawn from the 39 |
| clustered SE of accuracy | 0.005 to 0.009 | conversation-level means |
| clustered SE of mean tIoU | 0.025 to 0.035 | conversation-level means |

The design effect from clustering is 1.18 to 1.54, so treating the 390 questions as independent
understates the standard error by 10 to 25 percent. An empirical demonstration of the size of the
sampling error is sitting in the results directory:
`qwen3.8-27b.units-joint-demo.large-v3-turbo.partial.json` is the same run after 21 of 39
conversations and scores 0.7954, against 0.7748 on the full 39. That is a 0.021 swing from nothing but
which conversations were counted.

It also helps to know the granularity of the metric. One annotated positive moving from tIoU 0 to
tIoU 1 is worth 0.6/195 = 0.0031 in score on the training set, plus 0.4/390 = 0.0010 for its accuracy
point, so 0.0041 in total. On a 190-question set it is 0.0063 + 0.0021 = 0.0084. The quoted 0.006
noise floor on training is therefore 1.9 positives, and the 0.010 validation floor is 1.6 positives.
Any claim that rests on a 0.006 to 0.014 difference rests on two to four questions.

**Severity: high.** This is the yardstick under which the whole day's decisions were made.

**What to do.** Report score plus or minus a clustered SE for every single run, and use the paired
cluster bootstrap for every comparison. Ten lines of code; the helper I used is a 30-line function
over the existing `questions` list in each result file. Retire the phrase "above the 0.006 noise
floor".

### 5. Which claims survive a paired test and which are inside the noise

Paired cluster bootstrap over the 39 conversations, 3,000 to 4,000 resamples, differences in score.
The corrected accounting of findings 1 and 2 is used where the two arms had different offsets.

Survives:

| comparison | difference | 95 percent CI | verdict |
|---|---|---|---|
| words vs units (Qwen3.8-27B, large-v3), entry 34 reading 5 | -0.0505 | [-0.0650, -0.0368] | supported, strongly |
| gpt-oss-120b vs Qwen3.8-27B (units-claim), entry 34 reading 3 | -0.0223 | [-0.0404, -0.0054] | supported |
| Qwen3.6-35B-A3B vs Qwen3.8-27B (units-claim), entry 34 reading 3 | -0.0196 | [-0.0359, -0.0058] | supported |
| 27B vs 4B (entries 34, 35: "the 27B lifts the score by 0.06 to 0.08") | +0.08 | far outside | supported |
| 4B joint-demo vs 4B units (entry 33), corrected | +0.0298 | [+0.0050, +0.0554] | supported |
| validation run C vs A, +0.051 (entry 20) | +0.051 | about 5 x the 0.010 validation SE | supported |
| turbo's end fingerprint, 49 percent within 20 ms vs at most 15 percent (entries 17, 29) | +34 pp | SE about 3.6 pp | supported, about 9 SE |
| hard spans shared across models (entry 36) | see finding 9 | 2.0 to 2.4 x chance | supported |

Presented as a gain but inside the noise:

| comparison | difference | 95 percent CI or context | verdict |
|---|---|---|---|
| entry 34: few-shot pays on Qwen3.8, 0.762 to 0.780 | +0.0146 corrected | [+0.0004, +0.0299], p about 0.04 | borderline, dies under multiplicity (finding 11) |
| entry 34: few-shot +0.009 on Qwen3.6 | +0.009 | paired sd 0.009 | inside noise |
| entry 34: Qwen3.8 above Qwen3.6 (0.780 vs 0.774) | +0.0005 corrected | [-0.0155, +0.0171] | no difference at all |
| entry 35: joint-demo adds 0.014 on top of joint | +0.0139 (4B), +0.0087 (27B) | 4B [-0.0078, +0.0347] | inside noise |
| entry 35: few-shot 0.780 vs joint-demo 0.775 | +0.0038 corrected | [-0.0130, +0.0213] | correctly called inside noise by the log |
| entry 34 reading 4: turbo beats large-v3 by about 0.01 at the LLM level | +0.0058 as run, +0.0177 at each model's own offsets | as run [-0.0132, +0.0309] | the as-run comparison is inside noise; the offset-corrected one is not, see finding 6 |
| entry 37: Opus beats the 27B by 0.011 on the clean subset, "about two noise floors" | +0.0112 | [-0.0225, +0.0433] on 19 conversations | inside noise, and the CI is three times the effect |
| entry 36: "on the spans, Sonnet and the 27B are the same" (0.621 vs 0.628) | -0.007 tIoU | clustered tIoU SE 0.027 each | the claim of equality is unfalsifiable at this n, not the same as evidence of equality |
| entry 22: quote anchoring +0.023 locally | +0.023 | paired sd about 0.009 | probably real, but it did not transfer (entry 24), and 0.023 is only 2.5 paired SE |
| entry 30: "net effect within the 0.006 noise for units" | see finding 7 | sign reverses after the offset fix | wrong conclusion, not merely noisy |
| entry 13, 14: 0.5869 then 0.6086 | +0.0217 | validation SE about 0.022 to 0.030 per run, 0.010 run to run on identical code | mixes two different sources of variation |
| entry 24: run D 0.6632 vs run C 0.6599 | +0.0033 | correctly called inside noise | correct |
| entry 15: A minus B 0.0024 | see finding 1 | expected effect 0.0169, test power 1.5 SE | underpowered, and the conclusion is wrong |

The log's own hygiene is good in two places and should be kept as the house style. Entry 25 reports
that the leave-one-conversation-out threshold choice scores 0.699 out of sample against 0.702 as
answered, that is, it reports the honest out-of-sample number even though it is worse. Entry 29
flags its own in-sample offset fit. Both are the right instinct.

### 6. The ASR ranking is right, but for the wrong reason, and Parakeet is closer than the log says

**Claim in the log.** Entry 34 reading 4: "turbo beats large-v3 by about 0.01 and Parakeet by 0.03 to
0.04 for the same model and prompt". Entry 29's table ranks twelve configurations by an oracle ceiling
with per-model in-sample offsets separated by 0.01 to 0.03.

**Evidence.** All three LLM-level ASR rows were run at large-v3's offsets, which handicaps turbo and
badly handicaps Parakeet (entry 29's own fit gives Parakeet an end offset of -0.36, so -0.14/+0.12 is
0.48 s wrong at the end). Re-scoring `qwen3.8-27b.units.*` at each transcript's own best offsets, under
the live spans-on-no policy:

| transcripts | as run (-0.14/+0.12) | at its own best offsets |
|---|---|---|
| large-v3-turbo | 0.7718 | 0.7832 (-0.22/+0.00) |
| large-v3 | 0.7655 | 0.7655 (-0.14/+0.12) |
| parakeet-tdt-0.6b-v2 | 0.7307 | 0.7598 (-0.22/-0.28) |

So turbo's advantage over large-v3 is +0.018 rather than +0.006, and its advantage over Parakeet is
+0.023 rather than +0.033. Parakeet at an effective 0.76 with an RTF of 0.004 against turbo's 0.043
is a real option for the 60 s budget, not the also-ran the log implies.

The entry 29 ceiling column cannot separate 0.848 from 0.838 from 0.834: the mean of 195 oracle tIoUs
has a standard error of roughly 0.014, so the top four rows are one standard error apart, and the
winner of a maximum over twelve in-sample-fitted rows carries a selection bias on top of that. The
turbo decision is nonetheless safe, because it rests on a much sharper statistic: 49 percent of
annotated ends within 20 ms of a turbo word end against at most 15 percent for any other model, a
34 percentage point gap with a standard error of about 3.6 points.

**Severity: medium.** The decision is unaffected; the stated evidence is not the evidence that
supports it, and a fast-ASR fallback was dismissed on a biased comparison.

**What to do.** In entry 29, drop the ceiling ranking as the stated reason and lead with the end
fingerprint. Re-derive the LLM-level ASR comparison at per-tag offsets (arithmetic, no runs needed).
Keep Parakeet as the documented fallback if the H100 serving latency turns out tight.

### 7. Entry 30's conclusion reverses sign once the offsets are matched

**Claim in the log.** Entry 30: "Few-shot examples of the annotators' spans do not help qwen3:4b ...
Net effect within the 0.006 noise for units, a loss for words."

**Evidence.** The baseline row in entry 30 (accuracy 0.967, tIoU 0.522, score 0.700) is
`qwen3-4b.units-pyes.large-v3-turbo.json` at 17:32 with offsets -0.20/-0.02.
`qwen3-4b.units-fewshot.large-v3-turbo.json` at 19:39 has offsets -0.14/+0.12. Correcting the few-shot
run to the baseline's offsets gives 0.7096 against 0.6996, a paired difference of +0.0101 with a
95 percent CI of [-0.0087, +0.0281]. That is not a demonstrated gain either, but it is no longer
"does not help", and it is consistent in sign and size with the +0.0146 the 27B shows. The same
correction turns entry 32's joint gain from +0.010 to +0.0205 and entry 33's joint-demo gain from
+0.024 to +0.0298.

**Severity: high**, because entry 30 is the reason few-shot was labelled "not served" and only kept
"for the 27B, where a model that can actually use the examples may behave differently". That
interpretation was built on an artifact.

**What to do.** Rewrite entry 30 with the corrected numbers and the paired CI. The honest conclusion
is that the 4B and the 27B respond to few-shot examples in the same direction by about +0.010 to
+0.015, and that neither measurement separates it from zero on 39 conversations.

### 8. Leave-one-out hygiene: the code is correct and the offsets are nearly unbiased

**What I checked.** `FewShot.set_conversation(stem, asr)` sets `self.exclude` and `examples()` filters
`e['stem'] != self.exclude`, which drops all ten questions of the tested conversation, not just the
tested question (`bench/llm/prompts.py`). `JointDemo.demos_for` filters the same way, and
`JointDemoFewShot.set_conversation` forwards to both. `bench/llm/bench.py` calls `set_conversation`
once per conversation before dispatching its questions, and `bench/llm/dump_prompts.py` does the same
in both `dump` and `score`. So the leave-one-conversation-out claim in entries 30, 33 and 34 holds as
written. I could not find a path where a conversation's own gold reaches its own prompt.

**Residual channels and their size.**

1. *Pool size.* Under LOCO the example pool is 38 conversations; at deployment it is 39. That is a
   2.6 percent advantage at deployment, so the LOCO estimate is very slightly pessimistic, not
   optimistic.
2. *Shared question templates.* Only 3 of 390 training questions are exact duplicates of another
   question, and all three pairs are in different conversations ("does the patient have a sore
   throat?", "was the patient given a pneumococcal vaccination?", "are there no signs of
   complications?"). The Jaccard of the nearest cross-conversation positive question has median 0.20,
   p90 0.50, and 16.4 percent of positives have a near-paraphrase elsewhere. The few-shot selector
   uses exactly that similarity, so a sixth of the examples shown are near-paraphrases of the tested
   question from a different conversation, with that conversation's marked stretch attached. This is
   legitimate (the training labels are available at deployment too) and the deployment condition is
   the same, so it does not inflate the LOCO estimate. It does mean the measured few-shot gain will
   not transfer to a test question with no near-paraphrase in the training set.
3. *Offsets fitted in sample.* `bench/asr/fit_edges.py` is genuinely LOCO (see its docstring and the
   `_loco` helpers), so entry 5 and entry 17 are clean. Entry 29 is explicitly in-sample and says so.
   I measured the size of that optimism directly: sweeping the offsets on the realised runs, the
   in-sample optimum beats turbo's LOCO rule by 0.0004 to 0.0010 in score. So the in-sample caveat in
   entry 29 is real but worth one thousandth, not a hundredth. Two median-based constants fitted on
   195 spans simply cannot overfit much: the standard error of the median start offset is about 16 ms,
   and 16 ms on a 2.88 s median span is about 0.005 in tIoU.
4. *Oracle unit selection.* The offsets in entries 5, 17 and 29 are fitted on the oracle-selected
   sentence merge, which is chosen by best tIoU against the gold. So they are the offsets that are
   optimal given the *right* unit, while the served model picks the wrong unit 18 to 27 percent of the
   time. Empirically this does not matter: the optimum on the realised runs is -0.22/+0.00 against
   the oracle-fitted -0.20/-0.02.

**Severity: low.** This is the part of the day that was done right, and it is worth saying so in the
log so nobody re-litigates it.

**What to do.** Nothing, except record in `bench/README.md` that the few-shot and demo variants are
LOCO by conversation and that this was audited, and record the measured 0.001 in-sample offset
optimism next to entry 29's caveat so it stops being a worry.

### 9. The validation gold spans are 1.12 s longer than the training gold spans

**Claim in the log.** Entry 2 characterises the gold span length distribution from the 195 training
spans (median 2.88 s). Entry 28 notes qualitatively that on validation "the annotators often include
the question that prompted the answer or the confirming reply". Every prompt instruction, few-shot
example and edge rule is calibrated to the training distribution.

**Evidence.** The 95 recovered validation spans in `bench/mine/span_state.json` against the 195
training spans in `data/question_train.csv`:

| statistic | training (n = 195) | validation (n = 95) |
|---|---|---|
| mean | 3.21 s | 4.33 s |
| sd | 1.91 s | 2.48 s |
| p10 / p25 / p50 / p75 / p90 | 1.32 / 1.94 / 2.88 / 4.04 / 5.44 | 1.90 / 2.64 / 3.90 / 4.80 / 7.72 |
| share under 2 s | 28.7 percent | 11.6 percent |
| share over 5 s | 14.9 percent | 24.2 percent |

Welch t = 3.86 on 150 degrees of freedom, p below 0.001. Clustering by conversation (39 against 19)
does not remove it: the cluster bootstrap of the mean difference gives +1.12 s with a 95 percent CI of
[+0.51, +1.72].

**Consequences.** The convention differs between the two annotated sets, so neither is a safe guide to
the third. Entry 30's observation that few-shot examples pushed the 4B's median span from 2.50 s to
2.92 s "against a gold median of 2.88 s" reads as convergence on the target, but the held-out target
median is 3.90 s, so the corrected reading is that the examples moved the model a third of the way. The
"too little" versus "too much" balance you tune on training (31 against 28 in entry 30) is not the
balance you will be scored on.

**Severity: high.** It bounds how much any further edge or granularity tuning on the training set can
be trusted, and it is the best available explanation for why local gains keep failing to transfer
(entries 24, 30).

**What to do.** Do not fit any further length or granularity rule to the training distribution alone.
Prefer configurations that are flat across a plus or minus 1 s widening of both edges, and record that
sensitivity next to each candidate. Compute the widening sensitivity on the training set only, so this
stays a robustness criterion rather than a validation fit. Also note that you cannot tell whether the
difference is an annotation convention or a question-mix difference, because the validation hand labels
carry no `question_type` and so the hard-negative to off-topic mix (142 to 53 on training) is unknown
for validation.

### 10. The "never tune on validation" rule was broken 24 minutes after it was written

**Claim in the log.** Entry 27: "the pipeline is never tuned on them". Entry 28: "Rule kept: the
validation labels are a held-out measurement only; the served pipeline is never tuned on them."

**Evidence.** `git log -S'confirming reply'` returns exactly two commits. The first is e75a2da at
19:08, which is findings log entry 28, where the phrase appears in the sentence "the annotators often
include the question that prompted the answer or the confirming reply", cited to validation samples 3,
44 and 80. The second is e297b77 at 19:32, which writes that same insight into the prompt itself, in
`bench/llm/prompts.py`, as `_FEWSHOT_NOTE`: "when the fact is completed by the question that prompted
it or by the confirming reply, the marked stretch includes those utterances too". That note is appended
to the system prompt of `units-fewshot`, `words-fewshot` and `units-joint-demo-fewshot`, which is the
configuration entry 34 selects to serve.

To be fair to the record, I checked the rest and the served prompt is clean: `model.py`'s `SYSTEM` was
last changed at 17:18, before the validation labels existed, and the "a question and its answer, a
statement and its number" phrasing in `JointDemo`'s system note is copied from that 17:18 version, not
from entry 28. The single contaminated string is `_FEWSHOT_NOTE`'s "confirming reply" clause.

**Why it matters more than one sentence should.** The consequence is not that the training-set gain of
+0.015 is inflated (the instruction pushes toward longer spans, which on the training distribution is
if anything a handicap, per finding 9). The consequence is that the validation set can no longer serve
as an independent check on the configuration you intend to serve, which was the whole point of
recovering the labels. Once the recovered spans shape the prompt, a validation run on that prompt
measures fit, not generalisation.

**Other risks on the same rule.** `bench/mine/span_state.json` and `bench/mine/agent_labels/*.json`
are tracked in git, so the validation gold is now one `csv.DictReader` away from
`FewShot.pool()`. `bench/mine/val_labels/` is untracked and not in `.gitignore`, so it will be picked
up by the next `git add -A`. And the 358-run probe has put a 1.0000 next to your name on the public
validation board (entry 31), which is a visible signal about method, whatever the rules say.

**Severity: high.** The loss is not points, it is the ability to measure.

**What to do.** Decide explicitly, in writing, which of two regimes you are in. Either the validation
set is held out, in which case revert `_FEWSHOT_NOTE` to wording derivable from the training spans
alone and keep the validation labels out of every pool; or the validation set is training data, in
which case say so, add all 19 conversations plus their recovered spans to the demo and few-shot pool
(that is 49 percent more labelled data and is probably worth more than every prompt tweak measured
today), and accept that you then have no held-out set at all and must rely on LOCO over 58
conversations for the final decision. The second regime is defensible and is probably the better play;
what is not defensible is the current position of having tuned on it while the log says you did not.

### 11. The Claude probe: the contamination caveat is unverifiable and no contamination effect is detectable

**Claim in the log.** Entry 37: 20 of 39 conversations were exposed because "each agent answered five
prompts in one context, and the worked examples inside prompt B carry the gold answers of conversation
A when A sits in the same batch", with the clean-19 table presented as "fair for every model".

**Evidence.**

1. *The exposure list cannot be reproduced.* `bench/results/probe/leaked.json` is a bare array of 20
   stems with no model key, no batch assignment and no generating script. I reconstructed the demo
   graph from `bench/results/probe/prompts/*.json` by matching each demo's transcript prefix against
   the conversations' own prompts, which recovers the two demos of all 39 conversations exactly. The
   leaked set is then not the set of conversations in a mutual demo pair (26 conversations, and 13 of
   them are absent from `leaked.json`), and it is not produced by batches of five in either numeric or
   filename order under either a symmetric or an order-respecting exposure rule (best match, 11 of 20).
   Seven of the flagged conversations (4, 17, 33, 42, 50, 82, 84) are not in any mutual demo pair, so
   their exposure requires a specific batch grouping that is not recorded anywhere. There is also no
   record that Haiku, Sonnet and Opus were batched the same way, and all three were scored against the
   same list.
2. *No contamination effect is measurable.* The clean-versus-dirty split moves the two uncontaminated
   Qwen rows too, because the 20 flagged conversations are simply easier: qwen3:4b scores 0.7168 clean
   against 0.7311 dirty, and Qwen3.8-27B 0.7659 against 0.7833, both about -0.016. Taking that as the
   reference and forming the difference in differences:

   | model | clean minus dirty | contamination effect (DiD against the 27B) |
   |---|---|---|
   | Haiku 4.5 | +0.0109 | +0.028 plus or minus 0.046 |
   | Sonnet 5 | -0.0686 | -0.051 plus or minus 0.057 |
   | Opus 5 | -0.0307 | -0.013 plus or minus 0.025 |

   None is distinguishable from zero, and Haiku's point estimate has the wrong sign: Haiku did better
   on the conversations the caveat says were contaminated. If the leak had helped, all three should
   have moved the same way.
3. *The clean subset halves the power for no measured benefit.* On the clean 19 the paired difference
   Opus minus Qwen3.8-27B is +0.0112 with a 95 percent CI of [-0.0225, +0.0433]. The log's "about two
   noise floors" compares +0.011 to the 0.006 same-code figure, which is the wrong denominator by a
   factor of three.
4. *A second, unmentioned exposure channel.* Even with no demo overlap, five conversations answered in
   one context means the model sees its own earlier answers, so the five outcomes in a batch are not
   independent and the conversation-level bootstrap understates the SE for the Claude rows. More
   importantly, the Claude rows were produced under a batched, shared-context protocol while the Qwen
   rows were one request per conversation, so the comparison confounds model with protocol.
5. *Opus's perfect accuracy is not distinguishable from the 27B's.* 190 of 190 correct puts a
   one-sided 95 percent upper bound of 3/190 = 1.6 percent on the true error rate (rule of three). The
   27B's 2 errors in 390 is 0.5 percent. There is no accuracy difference to speak of.

**Severity: medium to high.** The conclusion drawn ("the curve is flat-to-slightly-rising past the
27B; a frontier model buys about 0.01") happens to be the right conclusion, but nothing in the probe
establishes it, and the clean-subset table is not the clean comparison it is presented as.

**How to redo it.** One agent per conversation, one prompt per context, 39 independent contexts per
model, which entry 37 already recommends. In addition: write the batch and agent assignment to a file
at dump time so exposure is derivable rather than asserted; run the Qwen rows through
`dump_prompts.py score` on the same answer files so the protocol is identical across models; and score
all models at turbo's own offsets under the spans-on-no policy. With 39 conversations the paired SE
against the 27B will still be about 0.009, so plan for a difference of 0.02 or more to be detectable
and do not expect to resolve 0.01. Note that `dump_prompts.py score` silently skips a conversation
with no answer file and reports accuracy over whatever remains, so if a redone probe loses files the
scores become non-comparable without any warning; make it fail loudly instead.

### 12. The oracle best-of-four is not a lever, and an implementable consensus recovers a sixth of it

**Claim in the log.** Entry 36: "the best of the four models per question would score a mean tIoU of
0.719 against 0.640 for the best single model, so the models disagree on a useful fraction of the hard
cases; a consensus or self-consistency step over several prompts is the lever to test next".

**Evidence.** I reproduce the oracle number (0.7077 for the four models of entry 36, 0.7324 with Opus
added, against a best single of 0.6280 and 0.6545 respectively, so a gain of about +0.079 either way).
But the oracle picks the best of four per question using the gold, so it is a maximum over four noisy
estimates and is upward biased by construction. The implementable version is a consensus. Taking, per
question, the span with the highest total IoU against the other models' spans (a medoid, no labels
used) gives:

| set | best single tIoU | oracle max | medoid consensus |
|---|---|---|---|
| 4B, Haiku, Sonnet, 27B | 0.6280 | 0.7077 | 0.6405 |
| the same plus Opus | 0.6545 | 0.7324 | 0.6636 |

So the consensus recovers +0.0125 and +0.0091 of tIoU, which is +0.0075 and +0.0055 in score, inside
the paired noise. The realisable fraction of the oracle gain is about one sixth.

Entry 36's other claim in the same paragraph is on much firmer ground. Of the 195 annotated positives,
the counts scoring below 0.5 are 86 (4B), 75 (Haiku), 70 (Sonnet), 70 (27B), 63 (Opus). Pairwise
overlaps run 51 to 58 against a chance expectation of 23 to 27, a ratio of 2.0 to 2.4; 52 positives
are below 0.5 for Sonnet, Haiku and the 27B together against an independence expectation of 9.7, and
43 are below 0.5 for all five models. The hard spans really are shared.

**Severity: medium.** It would cost you a day.

**What to do.** Do not build an ensemble on the strength of the oracle number. If you want to test
consensus, the medoid above is already computed from files you have; the answer is that it buys
nothing measurable. Spend the effort on the 43 to 52 spans that every model gets wrong, which is where
entry 36's first reading correctly points, and where the convention question of finding 9 lives.

### 13. Multiplicity

The day produced 62 result files in `bench/results/llm/` and the log presents roughly thirty
comparisons. At a nominal 5 percent you expect one or two spurious significant results from that many
tests. The only prompt-variant comparison that reaches p below 0.05 on the paired cluster bootstrap is
few-shot versus units on Qwen3.8-27B (p = 0.041 as run, corrected difference +0.0146 with the interval
just clearing zero). Under even a mild correction for the ten or so prompt comparisons in entries 30
to 35 the required threshold is p below 0.005, which it does not meet. The comparisons that survive
any correction are the four in the top half of finding 5's first table, all of which have effects of
0.02 or more.

**Severity: medium.** **What to do.** Keep a single pre-registered list of the comparisons that decide
the served configuration, at most three or four, and judge everything else as exploratory.

### 14. Run-to-run noise was never measured for the 27B, and every 27B number is a single run

`noise-run1.json` and `noise-run2.json` are qwen3:4b on Ollama, temperature 0, four slots, ten
conversations. That figure (8 differing spans of 100) is then used to judge single vLLM runs of a 27B
with ten parallel requests, a different model, a different server and a different batching regime.
vLLM's numerics at temperature 0 are not bitwise stable across batch compositions, so the 27B's
reproducibility could be better or worse and you do not know which. Every row of entries 34 and 35 is
n = 1.

**Severity: medium.** **What to do.** One repeat run of the intended final configuration on the 39
training conversations, next time the H100 is up. It costs a few minutes and it is the only way to
know whether the 0.015-sized differences you are chasing are even reproducible on that stack.

### 15. Smaller measurement points

1. *Entry 12's 0.657 is a spans-on-no number.* It was produced by `local_evaluator.py` end to end,
   which credits spans on no, and it is then compared in entry 13 against validation scores. Given
   finding 1 the comparison is now valid after all, which is worth noting explicitly since entry 15
   told you to distrust it.
2. *The category tables are not comparable across entries.* Entry 28 and entry 30 use six columns
   (exact, too little, too much, shifted, wrong place, missed) summing to 195. Entry 35 drops "too
   little" and "too much", so its rows sum to 136 and cannot be compared with entry 30's. Each count
   out of 195 has a binomial standard error of about 3.3, so paired changes below about 7 counts are
   noise; the "exact 64 to 46 to 51" sequence in entry 33 is inside that, while "shifted 25 to 56 to
   60" is not.
3. *Entry 21's percentages.* 39 / 27 / 22 / 11 percent over 185 positives carry standard errors of
   3.3 to 3.6 percentage points each, which is worth writing next to them since the same four numbers
   are used to argue where the loss is.
4. *Entry 2's arithmetic verified.* Being 0.5 s off at both edges over the actual training length
   distribution gives mean tIoU 0.664 and 1.0 s gives 0.447, matching the quoted 0.66 and 0.45, if
   "off at both edges" means a shift of the whole span. Read as a symmetric widening the numbers would
   be 0.718 and 0.573, so the entry should say which it means.
5. *Entry 1 verified exactly.* All 390 annotated edges are multiples of 0.02 s to within 4.6e-13.
6. *The span mining verifies itself.* All 190 recovered validation edges in
   `bench/mine/span_state.json` are exact multiples of 0.02 s, although `span_probe.py` only rounds to
   three decimals and never snaps to the grid. Combined with the 95 verify probes at IoU above 0.995
   and the full table scoring 1.0, the recovery is as close to certain as this kind of thing gets. Put
   the grid check in the log; it is a free independent confirmation and stronger than the verify probe.
7. *The label-free calibration check of entry 6 is now two for two.* Training is 195 of 390 and
   validation is 95 of 190, both exactly balanced. It is reasonable to treat a yes count near half as
   a sanity check on the final run.
8. *Entry 27's count arithmetic is right.* 0.4/190 = 0.0021053 per question, and a four-decimal score
   resolves single questions up to 190, since the smallest gap between adjacent counts is 0.0021.
9. *The leaderboard cluster is one big tie.* With a 19-conversation validation set the sampling
   standard error of a single team's score is 0.022 to 0.030 and the paired standard error between two
   similar systems on the same conversations is about 0.011 to 0.016. Entry 31's thirteen teams between
   0.7066 and 0.7887 therefore span about three to five paired standard errors end to end, and
   adjacent ranks are indistinguishable. The same applies to the final board, which is scored from one
   run per team: your final score will carry an irreducible sampling error of roughly 0.025, larger
   than every prompt gain measured today. Entry 31's arithmetic for what 0.79 requires (binaries 0.95,
   tIoU 0.68) is sound as a point estimate but should carry that interval.
10. *Two identical scores on the leaderboard.* Entry 31 reads Brew&Booze and Eirik Solberg sharing
    0.7457 to four decimals as evidence of a shared pipeline. Worth a number: the four-decimal grid
    around 0.745 has spacing 0.0001, and 37 teams over a plausible 0.3-wide range gives a birthday-
    problem collision probability of roughly 0.2 for at least one exact tie somewhere on the board. So
    one coincident pair is unremarkable; a third team at the same value would not be.

## The three things to fix first

1. **Re-establish which span policy the live scorer uses, from the dump, before anything else.**
   `request_dump/answers.jsonl` reconstructs to 0.67591704 under spans-on-no against the portal's
   0.6759 for run F, and to 0.6590 under nulls-on-no. If that holds up when you check which run the
   dump belongs to, entry 15 is wrong, `local_evaluator.py` is right, and every bench number of the
   day is 0.005 to 0.010 higher than you have been quoting. This costs no portal runs and it changes
   the level of every comparison and the value of the yes-threshold work.

2. **Fix the offset plumbing and re-derive the tables arithmetically.** Make
   `bench/llm/bench.py` set `ASR_MODEL` from `--asr` before importing `model`, and make `model.py`
   fail rather than fall back to large-v3's offsets for an unknown tag. Then re-score the stored spans
   at -0.20/-0.02 (a shift plus a clamp, no re-running). That adds 0.011 to 0.014 to every turbo row,
   reverses entry 30, and puts the corrected best training estimate at about 0.797 for Qwen3.8-27B
   few-shot and 0.807 for Opus.

3. **Decide, in writing, whether the validation set is held out or is training data, and put a
   clustered standard error on every score.** The `_FEWSHOT_NOTE` "confirming reply" clause came from
   entry 28's validation spans 24 minutes after the rule was written, so the current position is
   neither. Pick one: revert the wording and keep validation clean, or fold all 19 conversations and
   their recovered spans into the pools and switch to leave-one-conversation-out over 58 conversations
   as the only decision metric. Either way, retire the 0.006 noise floor and report score plus or
   minus a clustered standard error (0.014 to 0.021 on training) with a paired cluster bootstrap for
   every comparison, because the validation gold spans are 1.12 s longer than the training ones and
   the final attempt's own sampling error is about 0.025, which is larger than every gain in the log.
