# Edge ideas, ranked: what to build, what to run tonight, what to refuse

Written 2026-09-17, evening. For Elias and the team.

## What this page is

Six brainstormers worked the case in parallel, each through one lens. Three ran on Opus (lenses `opus-systems`, `opus-localization`, `opus-answering`) and three on Fable (lenses `fable-failure`, `fable-data`, `fable-wildcard`). Every brainstormer measured on the cached large-v3 transcripts and `data/question_train.csv` before proposing. Two judges, Opus and Fable, then scored every idea for gain and confidence and gave a verdict (`do-tonight`, `do-this-week`, `maybe`, `trap`). The pool held seventy-seven ideas. Most were duplicates of a dozen mechanisms. This page merges them.

Read `00-synthesis.md` first. It holds the task, the committee verdicts and the pipeline sketch. This page adds the ideas that came after, marks which were already in the synthesis, and turns the judged pool into an order of work.

Terms used below. tIoU is temporal intersection over union, the overlap of two spans divided by their union. ASR is automatic speech recognition. LLM is a large language model. LOCO is leave-one-conversation-out, the cross-validation that holds out all ten questions of one conversation at a time. RTF is real-time factor, seconds of compute per second of audio. IDF is inverse document frequency, the weighting that makes rare words count more in a lexical match. MAE is mean absolute error. SD is standard deviation. A/B means two runs that differ in exactly one setting. P(yes) is the probability the answerer assigns to yes. VAD is voice activity detection. CTC is connectionist temporal classification, the alignment mechanism behind wav2vec2-style aligners. vLLM and Ollama are the two LLM servers in play. LSF is the batch scheduler on the DTU cluster.

Two facts about our own repo that bear on everything below.

The only answering bench on disk (`bench/results/llm/qwen3-4b.*.large-v3.json`) is a two-conversation smoke run. Its rows are in the table. When a judge writes "the selector realises tIoU 0.49", that is this run. It is eleven spans and it is not a number yet. The full bench over the thirty-nine conversations runs tonight.

| smoke run (qwen3:4b, `units` variant, large-v3) | value |
|---|---|
| conversations | 2 |
| questions | 20 |
| positive accuracy | 11/13 |
| hard_negative accuracy | 4/4 |
| off_topic accuracy | 3/3 |
| mean tIoU over annotated yes | 0.415 |
| tIoU when answered yes (diagnostic) | 0.490 |
| no span returned | 2 |
| yes rate | 0.55 |
| LLM wall per conversation, worst | 8.1 s |

The judges' gain scale is in score points on the 0 to 1 scale.

| judge gain | meaning |
|---|---|
| 1 | up to 0.01 |
| 2 | 0.01 to 0.02 |
| 3 | 0.02 to 0.04 |
| 4 | 0.04 to 0.08 |
| 5 | over 0.08 |

Confidence runs 1 to 5. The ranking below sorts by gain times confidence, then by cost.

## The ten

| rank | idea | merged ids | lens of the best phrasing | in `00-synthesis.md`? | gain x confidence | verdicts (Opus, Fable) |
|---|---|---|---|---|---|---|
| 1 | Return a span for every question, including the ones answered no | 15, 28 | opus-answering | new | 9.0 | do-tonight, do-tonight |
| 2 | Quote-to-word anchoring, clamped to the cited unit | 44, 67, 18, 19, 17 | fable-failure | partly (quote field yes, edges from it no) | 9.0 | do-this-week, do-this-week |
| 3 | The validation set as a paired A/B oracle, with every request recorded | 74, 16, 49 | fable-wildcard | new | 8.0 | do-this-week, do-this-week |
| 4 | Fingerprint the annotators' timestamp tool; refuse energy snapping; lock the offsets to the ASR tag | 64, 69, 13, 27 | fable-data | yes (experiment 1) | 6.25 | do-this-week, do-this-week |
| 5 | Deadline ladder with a lexical rung instead of `(True, None)` | 0, 41, 21, 1, 20, 59, 72 | fable-failure | new (the synthesis kept the no-span guess) | 6.0 | do-tonight, do-this-week |
| 6 | Whisper decode hygiene, then refit the offsets | 43, 42, 27 | fable-failure | new | 6.0 | do-tonight, do-tonight |
| 7 | Clause units | 55 | fable-data | yes (experiment 2) | 6.0 | do-this-week, do-this-week |
| 8 | A real P(yes), then a yes threshold below one half | 30, 45, 54, 66, 29 | opus-answering | new | 5.0 | do-this-week, do-this-week |
| 9 | Pre-flight from outside the LAN, on a persistent VM | 48, 14, 10 | fable-failure | partly (experiment 4) | 5.0 | do-tonight, do-this-week |
| 10 | Extent selection: score the candidate merges, fix the unfitted gap rule | 57, 23, 46, 75, 65, 73 | opus-localization | new | 5.0 | do-this-week, maybe |

Ideas 1 and 8 are mutually exclusive in value. If the live grader scores spans next to a no, the asymmetry that idea 8 exploits disappears and the threshold goes back to one half. Idea 3 is the experiment that decides which of them is worth anything, and it needs idea 6 first so that two attempts with the same code produce the same transcripts.

### 1. Return a span for every question, including the ones answered no

Ids 15 (opus-localization) and 28 (opus-answering), found independently. New. Not in the synthesis, whose task paragraph repeats the README prose that a yes answered no scores zero on both halves.

**Mechanism.** The scorer that was ported from the evaluation service, `local_evaluator.Statistics.record` and `utils.mean_temporal_iou`, selects which questions enter the tIoU average by the gold label alone. It never reads the returned boolean. A span returned next to `false` on a question whose true answer is yes is scored in full. A span returned next to `false` on a true negative is discarded. So the two halves of the score decouple. A wrong no costs the accuracy point only, provided the span is there. The README prose says the opposite in one paragraph and, in the next, says a span next to a no can neither help nor hurt. Either way the downside is zero. `utils.validate_response` accepts a span beside `false`.

| quantity | value |
|---|---|
| `record('positive', label=1, prediction=0, gold=(10,13), predicted=(10,13))` | tIoU 1.000, accuracy 0.000 |
| `record('hard_negative', label=0, prediction=0, gold=None, predicted=(10,13))` | nothing appended |
| gain formula | 0.6 x (1 - positive recall) x mean tIoU of the spans attached to missed positives |
| Opus judge budget | +0.03 to +0.05 |
| author of 28, 15 to 20 percent missed positives at 0.65 tIoU | +0.03 to +0.07 |
| author of 15, at 60 percent recall and 0.60 tIoU | up to +0.14 |

**Decisive judge note.** Opus on 15.

> Verified in the shipped code, but the gain range is inflated: the spans you would attach to positives you MISSED are your worst spans, so budget +0.03 to +0.05, not +0.12.

Fable on 28.

> If the service is live on this the threshold ideas (30/54/66) become moot and 0.5 is right, so run the paired validation A/B before touching thresholds.

**Disagreement.** None on the verdict. Opus trims the gain range of 15; Fable accepts 28's range.

**First experiment.** Two validation attempts, byte-identical except that one sends the best-guess span next to every `false` and the other sends `null`. The difference divided by the per-span weight is the tIoU recovered. This needs idea 6 first (deterministic transcripts) and the endpoint of idea 9. The span attached to a no comes from idea 5's lexical rung, or from the LLM's cited unit when it cited one and still said no.

**Plugs into.** `model.py` `answer_all`, inner function `one()`, the line `span = span_from_ids(ids, units, duration) if yes else None`. The span is computed regardless of `yes`; when `ids` is empty the lexical fallback supplies it. The boolean does not change. Every volunteered span must be complete and ordered or `validate_response` raises (see the repair layer under "below the ten"). `api.py` and `example.py` are untouched. Bench: `bench/llm/bench.py` should score both ways, spans-on-no and nulls-on-no, from the same run.

### 2. Quote-to-word anchoring, clamped to the cited unit

Ids 44 (fable-failure, the canonical phrasing), 67 (fable-wildcard, the first-three-and-last-three-words output format), 18 (opus-localization, the clamp rules), 19 (opus-localization, grow-never-shrink as the tie-break) and 17 (opus-localization, two-branch offsets, self-declared null result at sentence level). Partly in the synthesis. Report 04 and the synthesis put a verbatim quote field in the schema so the model re-reads the detail, and `bench/llm/prompts.py` already has a `words` variant that asks for the first and last words and locates them with difflib. Using the quote to set the span edges instead of the unit ids is the new part.

**Mechanism.** `model.py` already asks the LLM for a verbatim quote and discards it except as a fallback when the model says yes and cites nothing. Normalise the quote and the word stream the same way (lowercase, strip punctuation, spell numbers and units one way so "47 millimoles per mole" and "47 mmol/mol" match). Fuzzy-align the quote to the word sequence and take the first and last matched word. Apply word-level edge rules, not the sentence-level constants (the start rule "end of the first word minus a constant" for sentence-initial edges, the raw word start for mid-sentence edges, the end constant on the end). Clamp the result to the cited unit. Never let it leave that unit. Never let it be shorter than the matched quote. Fall back to the unit span when fewer than three words match or the quote matches in two places. Resolve remaining ties outward, because one word too long costs about half of one word too short.

| candidate set, oracle selection, large-v3 | mean tIoU |
|---|---|
| sentence merges of up to four, constant offsets | 0.824 |
| clause merges, constant offsets | 0.875 to 0.882 |
| best contiguous word run, end +0.12 | 0.912 |
| best word run, -0.08 / +0.12 | 0.927 |
| best word run, -0.18 / +0.12 | 0.931 |
| best word run grown one word each side across a pause, unclamped | 0.650 |
| inside the correct sentence, whole sentence | 0.690 |
| inside the correct sentence, oracle sub-clause | 0.806 |
| optimal run, one word too long / two too long | 0.747 / 0.728 |
| optimal run, one word too short / two too short | 0.564 / 0.403 |

| expected gain | source |
|---|---|
| +0.03 to +0.05 score if the quotes are accurate on half the questions | 44 |
| +0.02 to +0.04 score | 67 |
| +0.03 realistic, +0.064 at the ceiling | 18 |

**Decisive judge note.** Opus on 44.

> I reproduced the ceiling myself, word runs at -0.08/+0.12 score 0.927 against 0.824 for sentence merges, and unlike the pure edge ideas this one also tightens the REALIZED span, which is where the 0.49-versus-0.824 loss actually lives.

Fable on 18.

> The 0.824->0.93 ceiling gap is the largest tIoU lever left, but the unclamped variant collapses to 0.650, so the clamp-to-cited-unit and fallback rules are the whole implementation.

**Disagreement.** None on the verdict. On the start rule, Fable prefers idea 68's single-parameter rule over idea 17's two pause buckets, whose bucket sizes are small.

| start rule on the sentence-initial gold starts | MAE |
|---|---|
| first word start + 0.36 | 0.117 s |
| first word end - 0.14 | 0.089 s |
| ridge on pause, first word duration, gap to second word, LOCO | 0.089 s |
| pause bucket sizes behind idea 17's two constants | n = 9 and n = 20 |

**First experiment.** Idea 67's design, endorsed by both judges. No LLM. For each of the annotated spans, take the gold span's own words from the large-v3 transcript as a perfect quote, align it back to the word stream, apply the word-level rules, clamp to the containing sentence and score. Expect the within-sentence oracle from the table. Then degrade the stand-in quote by dropping and adding one and two words at each end and re-score, which gives the sensitivity curve before any model output is trusted. Then, once tonight's full bench lands, re-score every question's returned `quote` against the unit-id span.

**Plugs into.** `model.py`: a new `span_from_quote(quote, ids, units, words, duration)` next to `span_from_ids`, called from `one()` when the quote matches, with `span_from_ids` as the fallback. The word stream must survive to that point (it does; `answer_all` holds `words`). `bench/llm/prompts.py`: the `words` variant already carries the difflib locator; add a `quote-anchor` variant that keeps the `units` output format and post-processes the `quote` field. `span_ceiling.py` gets a word-run mode so the oracle is reproducible from the repo.

### 3. The validation set as a paired A/B oracle, with every request recorded

Ids 74 (fable-wildcard, the paired design and the arithmetic), 16 (opus-localization, probe the live grader), 49 (fable-failure, record every request). New.

**Mechanism.** Validation attempts are unlimited and report a score to three decimals. One flipped answer moves the score by a fixed amount that is visible at that precision. Two attempts that differ in exactly one policy measure that policy on held-out audio with the real harness and the real network. One policy per pair. Do not try to extract labels by differencing; policy-level only. Alongside, the server writes each request's audio, questions, transcript with word probabilities, per-question LLM output and timings to a folder, so the hidden distribution can be audited without labels: yes count per conversation, word-probability holes, latency, garbage characters.

| quantity | value |
|---|---|
| score move per flipped answer | 0.4 / 190 = 0.0021 |
| score move per span | 0.6 / 190 x tIoU = 0.0032 x tIoU |
| wall time per attempt | 19 requests at about 30 s, about 10 min |
| validation score SD from conversation sampling alone | 0.027 |
| evaluation score SD | 0.019 |
| yes per conversation in training | 3 to 7 |

**Expected gain.** Protects against a loss of two to four points from a prior that fit the thirty-nine training conversations only, and it is the only honest test of ideas 1, 8 and the tag prior. Pairing cancels the sampling SD, which unpaired validation runs cannot.

**Decisive judge note.** Fable on 74.

> A paired A/B on the same 19 conversations cancels the 2.7-point sampling SD, so one toggle per attempt at ~10 min each is the only honest test of the risky priors and of the 28 loophole.

Opus on 16.

> Right instinct, one broken probe: you have no gold spans for the hidden validation set so probe (b) is impossible - the exact measurement is the byte-identical A/B of idea 28, and it only reads cleanly once idea 43 has made the pipeline deterministic.

**Disagreement.** On 49 the verdicts split (Opus do-this-week, Fable do-tonight). On the number of validation runs, idea 62 says cap at about five and idea 74 says one pair per risky policy. The judges accept both; the reconciliation is a shared log with one row per attempt and a rule that only a config that beats local by a clear margin gets a slot, except for the pairs listed here.

**First experiment.** The all-true, all-null body first, which should print the floor and confirm the arithmetic matches. Then the idea 1 pair. Then the tag-rule pair. Then the threshold pair, only if the idea 1 pair came back flat.

**Plugs into.** `model.py`: the policies to toggle become environment variables read at import (`SPAN_ON_NO`, `TAG_PRIOR`, `YES_THRESHOLD`). `example.py` `predict`: a request dump under a gitignored `requests/` folder, rotated so the disk does not fill during the evaluation run. Bench: `bench/results/validation_log.md`, one row per attempt with the git hash, the toggles, the score, the worst latency and the yes count.

### 4. Fingerprint the annotators' timestamp tool; refuse energy snapping; lock the offsets to the ASR tag

Ids 64 (fable-data), 69 (fable-wildcard, the refusal half), 13 (opus-systems, per-span forced alignment, subsumed), 27 (opus-localization, offsets as a configuration lock). Already in the synthesis as experiment 1. The judges' addition is the refusal of the whole waveform-edge family and the instruction to treat the fitted constants as tied to one decoder configuration.

**Mechanism.** All annotated timestamps sit on a twenty-millisecond grid. Gold starts sit after the acoustic speech onset and gold ends before the acoustic offset, which is the late-start, early-end fingerprint of a CTC aligner, not of a human on a waveform. Run every timestamped ASR and aligner the bench has over the thirty-nine files and, per tool, histogram the signed offset from each gold edge to the nearest word boundary. A tool whose residuals cluster at zero with small spread is the generator, and its timestamps replace the fitted constants. If none matches, keep large-v3 and the constants. Either way, make the offset fit a one-command step keyed by ASR tag, because idea 6 changes the decoder and `models/parakeet-*` is already downloaded, so a silent swap is one command away.

| edge estimator | MAE against gold |
|---|---|
| Whisper first word start + 0.36 (start) | 0.117 s |
| Whisper first word end - 0.14 (start) | 0.089 s |
| Silero VAD, debiased (start) | 0.157 s |
| RMS energy onset, 15 dB above floor (start) | 0.298 s |
| Whisper word end + 0.12 (end) | 0.080 to 0.086 s |
| acoustic offset (end) | 0.236 s |
| clause ceiling with both edges energy-snapped | 0.786 (vs 0.875) |
| match threshold for a tool | residual MAE under 0.05 s |
| no-gain threshold | residual MAE above 0.09 s |
| timestamps on the 0.02 s grid | 195/195 spans |
| timestamps that also fit Parakeet's 0.08 s grid | 15/195 |

**Expected gain.** One to three points if a tool matches; zero if none does; the refusal half saves an evening either way.

**Decisive judge note.** Fable on 64.

> Gold starts ~0.3 s after acoustic onset and ends ~0.2 s before offset is the late-start/early-end fingerprint of a CTC aligner (WhisperX/MMS); the runners exist, so this is the first H100 job.

Opus on 64.

> Calibrate expectations: with the realized tIoU at 0.49 against an 0.824 ceiling, a perfect aligner mostly improves spans that already score 0.96.

**Disagreement.** Priority, not substance. Fable puts it first on the H100 because the job exists. Opus says the edge is not where the score leaks and ranks it as a cheap parallel sweep. Both agree it runs tonight. On 13 both say maybe; Fable subsumes it into 64 (if whole-transcript MMS edges do not beat the constants, per-span crops will not either).

**First experiment.** Submit `bench/hpc/asr_bench.lsf`; read `bench/results/asr/summary.md` table A (start and end medians per tag) and table B (distance to the nearest word boundary). Add a per-tag residual MAE column to `compare.py` if it is not there.

**Plugs into.** `bench/asr/compare.py` already emits a fitted offset pair per tag. `model.py` reads `START_OFFSET` and `END_OFFSET` from the environment; replace with an `offsets.json` keyed by `ASR_MODEL` tag, written by `span_ceiling.py --fit`, so a decoder change without a refit fails loudly.

### 5. Deadline ladder with a lexical rung instead of `(True, None)`

Ids 0 (opus-systems, the slot and watchdog), 41 (fable-failure, the most concretely specified, with the warning that Python threads cannot be killed), 21 (opus-localization), 1 (opus-systems, replace the except branch), 20 (opus-localization, the lexical localizer), 59 rung a (fable-data), 72 (fable-wildcard, the safety net half). New. The synthesis' pipeline ends with "on any exception return a guess with no span", which is exactly the fallback this replaces.

**Mechanism.** Stamp the arrival time at the POST. Hold a mutable slot with the best legal response so far. Tier 0 is all-yes, all-null the instant the body parses. Tier 1 is the lexical answer and span for all ten questions the instant the transcript exists (IDF-weighted content-word overlap over the sentence units, windows of up to three units with length normalisation, the fitted offsets; a threshold on coverage for yes). Tier 2 is the LLM answer, written into the slot as each question finishes. A watchdog thread returns the slot at the hard deadline whatever the pipeline is doing. Replace the per-question except branch, which returns `(True, None)`, with the tier 1 answer. Keep the last-resort all-yes-no-span rung only for a decode failure. A thread that is still on the GPU when the watchdog fires will run into the next request, so the next request must see a busy GPU and shorten its own budget. Never let the lexical span override a confident LLM citation; it is a bad primary localizer.

| quantity | value |
|---|---|
| current except branch `(True, None)`, expected score per question | 0.200 |
| lexical rung, always yes, mean tIoU over positives | 0.521 (score 0.512) |
| lexical rung, tuned yes threshold, in-sample | 0.578 |
| lexical rung, coverage threshold 0.34 (fable-failure's variant) | accuracy 0.736, tIoU 0.394, score 0.531 |
| lexical logistic prior, LOCO accuracy | 0.759 (off_topic 53/53) |
| whole lexical system, LOCO | 0.585 |
| MiniLM bi-encoder over the same candidates | 0.400 tIoU, 198 ms per conversation |
| lexical scorer cost | 0.48 ms per question |
| gold sentence hit by the lexical argmax | 128/195 |
| worst case in `model.py` today | 39 s ASR + 25 s LLM_TIMEOUT = 64 s |
| bootstrap, laptop RTF profile, P(at least one timeout per attempt) | 0.37 |
| bootstrap, P(five in a row) | 0.000 |
| cost of one lost conversation | 10/380 of the attempt |

**Expected gain.** Small in expectation on the happy path, up to the whole tail on a bad day. Per question that hits a failure path the rung is worth about 0.35 more than the current guess. The `LLM_TIMEOUT` in `model.py` is applied to ten concurrent calls against fewer Ollama slots, so it is a queue-position timeout, and tail questions hitting it is not hypothetical.

**Decisive judge note.** Opus on 0.

> model.py has no global deadline at all (39 s worst ASR + 25 s LLM_TIMEOUT = 64 s > 60), so this is the one build that makes the attempt structurally survivable - but build it ONCE: ideas 1, 21, 41, 59a and 72 are the same code and must not be built five times.

Fable on 41.

> Same ladder as 0/21 with the right fallback rung; the 64 s worst case is real in model.py today but the ASR part is the fallback tail that 43 removes.

**Disagreement.** Timing. Opus says do-tonight for the ladder. Fable says do-this-week, after idea 6 has removed the ASR tail and after the lexical rung (ideas 1 and 20, do-tonight for both judges) exists. Both judges reject the off-topic short-circuit in 59 and 72 and the MiniLM retriever in 72.

**First experiment.** Drop the lexical scorer into `model.py` as `fallback_answer(units, question) -> (bool, span)`. Score it alone on the training questions (expect the always-yes row of the table). Run `local_evaluator.py` with `LLM_TIMEOUT=0.5` so every question fails and confirm the score moves from the floor to about the lexical row with zero failed conversations. Then add `FAKE_ASR_DELAY` so five of thirty-nine conversations would blow the budget and confirm zero timeouts in the report.

**Plugs into.** `model.py` `answer_all` (slot, watchdog thread, deadline from the environment), `one()` except branch (line 288 to 290), a new `lexical.py` or a block in `model.py` with the tokenizer, stoplist and IDF table built per transcript. Idea 1 uses the same function for the span on a no. Bench: a `lexical` variant in `bench/llm/prompts.py` that never calls the LLM, so the rung's score is tracked on every bench run.

### 6. Whisper decode hygiene, then refit the offsets

Ids 43 (fable-failure), 42 (fable-failure, confidence-gated window repair, conditional), 27 (refit). New.

**Mechanism.** faster-whisper's default temperature ladder re-decodes a thirty-second window with sampling whenever the compression ratio or log-probability threshold trips. That is the latency spike and the source of run-to-run non-determinism. Set a single temperature of zero, `condition_on_previous_text=False` (stops the drift into Korean tokens after a silence), `hallucination_silence_threshold=2`, and measure `vad_filter` as a separate toggle because it changes timestamps. Then refit the two edge offsets on the new transcripts. The three slowest files are exactly the files with repetition loops or garbage, which is the fallback path. Build idea 42's window repair only if the request dumps from idea 3 show a hole like sample_81 on the hidden audio.

| quantity | value |
|---|---|
| large-v3 RTF on the laptop, mean / p50 / p90 / max | 0.113 / 0.092 / 0.218 / 0.397 |
| ASR wall per conversation, mean / max | 13.6 s / 39.1 s |
| slowest files | sample_43, sample_57, sample_81 |
| words with p under 0.35 in sample_81 / sample_57 | 78 / 77 |
| sample_81 hole (no words emitted) | 60.0 s to 68.9 s |
| positives of sample_81 inside the hole | 3 of 4 |
| files with hallucinated non-Latin tokens | 1 of 39 |
| words with p under 0.5 across all files | 2.8 percent, nearly all function words |
| gold spans shorter than 1.3 s that a VAD cut could clip | 10 |

**Expected gain.** Zero to one point direct. Worst-case ASR from thirty-nine seconds toward fifteen. Deterministic transcripts, which idea 3 silently depends on.

**Decisive judge note.** Opus on 43.

> Best cost-to-value on the whole list: it kills the RTF tail (the three slow files are exactly the fallback files), and it makes runs deterministic, which every validation A/B on this list silently depends on; budget an hour after it to refit the two offsets.

Fable on 42.

> One catastrophic file in 39 (~0.004 expected on eval); do 43 first and only build the window repair if 49's validation dumps show sample_81-type holes on the hidden audio.

**Disagreement.** None on 43. On 42 Opus says do-this-week, Fable says maybe.

**First experiment.** Re-transcribe all thirty-nine with the new settings under a distinct tag, print per-file seconds and the max, run `span_ceiling.py` on the new tag, diff word counts against the current cache, and check the ten short gold spans if `vad_filter` is on.

**Plugs into.** `model.py` `transcribe()`, the `model.transcribe(...)` call (line 117 to 120). `transcribe_cache.py` and `bench/asr/run_faster_whisper.py` need `--temperature` and `--no-condition` flags and a tag suffix (`+clean`) so the variant never overwrites `large-v3`. `bench/asr/compare.py` refits per tag.

### 7. Clause units

Id 55 (fable-data). Already in the synthesis as experiment 2.

**Mechanism.** Split the word stream at commas, semicolons and colons as well as terminal punctuation and long pauses, and let the selector cite a first and a last clause id. Render as `[12a] ... [12b]` under the sentence number so the id list the model sees does not double in size. A share of gold ends fall mid-sentence, and of those about half sit exactly on a comma.

| candidate set | ceiling | spans under 0.5 |
|---|---|---|
| sentence merges of up to four, offsets | 0.824 | 23 |
| clause merges of up to six, offsets | 0.882 | 11 |
| any word run | 0.916 | |
| gold ends not on a sentence-final word | 37/195 | of which on a comma 17 |
| best clause merge uses 1 / 2 / 3 to 6 units | 110 / 45 / 39 | |
| gold spans shorter than half a sentence | 20/195 | |

**Expected gain.** Two to three and a half points at the ceiling; half of that realistically.

**Decisive judge note.** Fable on 55.

> +0.058 tIoU ceiling from a deterministic regex change, cheaper and safer than quote trimming; render as [12a]/[12b] so the id-citing task does not double in size.

Opus on 55.

> That is ceiling not score - the selector currently realizes 0.49, so raising a ceiling it is 0.33 below only pays if the trimming ideas land, and more, shorter units make id-citing harder for a 4B model.

**Disagreement.** Real, on ordering. Fable ranks clause units above quote anchoring because a regex cannot collapse to 0.650. Opus says the ceiling only pays once the realised span is close to it. They are complementary: idea 2 clamps the quote to the cited unit, and a smaller unit makes that clamp tighter.

**First experiment.** Change `make_units` behind an environment flag, rerun `span_ceiling.py` on the cached transcripts with the clause merge and confirm the ceiling row, then rerun the bench on the same transcripts and compare mean tIoU and the under-0.5 count against the sentence-unit run with the same model.

**Plugs into.** `model.py` `make_units` (line 146 to 166), `_TERMINAL`, `render_transcript`. `span_ceiling.py` `sentences()` and `merges()` with a larger `--k`. Bench: the same `units` variant, run with the flag set and `--suffix .clause`.

### 8. A real P(yes), then a yes threshold below one half

Ids 30 (opus-answering, canonical), 45 (fable-failure), 54 (fable-data), 66 (fable-wildcard), and the enabler 29 (opus-answering). New.

**Mechanism.** If spans only count on a yes, a missed positive loses the accuracy point and the whole tIoU term, while a false yes loses the accuracy point only. The per-question weights make the ratio exactly three. The break-even probability for saying yes is well below one half. To use it the pipeline needs a probability, which the current JSON string does not give. On vLLM, put `answer` first in the schema and read `top_logprobs` at that token, then fit a one-parameter Platt scaling LOCO on the training questions. On Ollama, logprobs are not exposed in every build; the alternative is a small self-consistency vote at a positive temperature, which costs several calls. Sit at a threshold of 0.35 until idea 3 settles idea 1, because the loss surface is flat there under both regimes.

| quantity | value |
|---|---|
| accuracy weight per question, evaluation set | 0.4 / 380 = 0.001053 |
| tIoU weight per annotated yes | 0.6 / 190 = 0.003158 |
| break-even P(yes), q = expected tIoU when right | 1 / (2 + 3q) |
| break-even at q 0.70 / 0.60 / 0.50 | 0.244 / 0.263 / 0.286 |
| Monte Carlo, 82 percent model, threshold 0.50 / 0.35 / 0.30 / 0.26 | 0.732 / 0.750 / 0.750 / 0.747 |
| Monte Carlo, 75 percent model, threshold 0.50 / 0.30 | 0.678 / 0.707 |
| gain over 0.5 at 82 / 75 / 88 percent accuracy | +0.018 / +0.029 / +0.007 |
| gain if the live grader scores spans on a no | about -0.01 |

**Decisive judge note.** Opus on 30.

> Best-argued of the four threshold duplicates and the only one that spots the killer interaction - if the loophole in idea 28 is live the asymmetry vanishes and 0.5 is optimal again, so sit at 0.35 until the validation A/B settles it and do not bank both gains.

Fable on 29.

> Enabler with no standalone gain; on vLLM put answer first in the schema and read top_logprobs at that token, then Platt-scale on the 390 LOCO.

**Disagreement.** On 54 and 66 the verdicts split (Opus maybe as duplicates, Fable do-this-week). No disagreement on 30 or 29. Both judges say the gain shrinks as the answerer improves and vanishes if idea 1 is live.

**First experiment.** Rerun the bench with logprobs recorded, dump P(yes) per question, plot a reliability diagram in ten bins, then sweep the threshold from 0.15 to 0.60 LOCO on the full score with the real spans the pipeline produced, not an assumed q.

**Plugs into.** `bench/llm/bench.py` `Client.body` (add `logprobs` and `top_logprobs`) and `run_question` (record `p_yes`). `model.py` `ask_llm` and `SCHEMA` (answer first), a `YES_THRESHOLD` environment variable read in `one()`. The Ollama path in `model.py` cannot do this cleanly; the serving-stack decision (vLLM on the cloud GPU) decides whether this idea is implementable in production.

### 9. Pre-flight from outside the LAN, on a persistent VM

Ids 48 (fable-failure), 14 (opus-systems, do not serve from the batch cluster), 10 (opus-systems, calibrate the deadline against a remote clock, folded into 48). Partly in the synthesis as experiment 4 (cloud VM, request limit above five megabytes, warm at import). New here: the checklist itself, the from-outside rule, and the refusal of the active-active build.

**Mechanism.** A script hits the submitted URL exactly as the evaluator does, from a second network, with the largest training body, and asserts: status 200; the proxy body limit is above the largest body (nginx's default returns 413 on every request); the proxy read timeout is above the request budget; the LLM sits fully on the GPU (`ollama ps` shows no CPU share); VRAM headroom after ASR and LLM warm-up; no filename-keyed transcript cache (hidden filenames fill the gaps in the training ids and would silently hit the wrong file); a process supervisor that restarts on crash; sleep and power management off; logs not filling the disk. Then run `local_evaluator.py --url <public>` three times and compare score, worst latency and yes count for drift. The endpoint must be a persistent VM; an LSF job is VPN-only, has a wall limit and can be pre-empted. Skip the second worker and the second region.

| quantity | value |
|---|---|
| largest MP3 / its base64 body | 3.71 MB / 4.95 MB |
| nginx default `client_max_body_size` | 1 MB |
| nginx default `proxy_read_timeout` | 60 s |
| body transfer at 1 Gbit/s / 100 Mbit/s / 10 Mbit/s | 0.04 s / 0.40 s / 3.96 s |
| server-side body handling (json.loads + b64decode) | 9.7 ms |
| training ids used in 1..95 / unused | 39 / 56 |
| hidden conversations (validation + evaluation) | 19 + 38 = 57 |
| large-v3 fp16 + 4B Q4 model + parallel KV on the 8 GB laptop card | on the edge |

**Expected gain.** Protects the whole attempt. A wrong proxy limit or a CPU-offloaded LLM costs everything, not a few points.

**Decisive judge note.** Opus on 48.

> Largest body is 3.71 MB of MP3 = 4.95 MB base64 against nginx's 1 MB default, which would 413 every single request and cost the entire attempt; two hours of checklist protects more score than any modelling idea here.

Opus on 14.

> The 'tunnel may breach no-cloud' worry is a misread: the rule bans hosted models in the request path, not your own VM, so skip the 6 h active-active build and just stand up one persistent box.

**Disagreement.** On 48, Opus do-tonight, Fable do-this-week. On 10, Opus do-this-week, Fable maybe (fold the one remote round-trip measurement into 48).

**First experiment.** `curl -X POST` the largest training body at the public URL from a phone hotspot; read `ollama ps` and `nvidia-smi` during the request.

**Plugs into.** A new `bench/preflight.py` (or shell script) beside `local_evaluator.py`; the VM's proxy and supervisor config; `api.py` untouched. `example.py` already catches a whole-conversation exception.

### 10. Extent selection: score the candidate merges, fix the unfitted gap rule

Ids 57 (fable-data, learned re-ranker), 23 (opus-localization, score the window), 46 (fable-failure, tighten `span_from_ids`), 75 (fable-wildcard, span policy), 65 (fable-data, second LLM pass on the extent), 73 (fable-wildcard, crop-and-refine cascade). New.

**Mechanism.** Three versions of one decision, how many units to return. The cheap version (46 and 75, one hour): `span_from_ids` extends the cited run while the next id is within two, which swallows an uncited unit under a policy nobody fitted. Tighten it to adjacency, add a soft length guard that only trims units the model did not cite, keep no padding, and never append the patient's backchannel. The fitted version (57 and 23): generate candidate windows of one to four units around the cited ids, score each with a handful of features (contains the cited ids, length inside the gold band, ends at a pause or sentence end, starts after a pause, outside the greeting and the goodbye, quote overlap), fit LOCO on the annotated spans, freeze. The expensive version (65 and 73): a second LLM call per yes that sees only the candidate merges and picks the tightest one that still contains every detail. Pick one of the three. Measure the right-neighbourhood rate first, because no extent rule saves a citation of the wrong sentence.

| quantity | value |
|---|---|
| oracle single sentence with offsets | 0.694 |
| oracle merge of up to four | 0.824 |
| oracle clause merge | 0.882 |
| retriever pick, single unit / always merge the next unit | 0.498 / 0.342 |
| retriever, length-normalised windows of up to three | 0.501 to 0.520 |
| best merge is 1 / 2 / 3 / 4 sentences | 140 / 39 / 6 / 10 |
| gold ends followed by a pause over 0.4 s | 138/189 |
| gold starts preceded by one | 107/189 |
| gold length inside [1.3, 6.0] s | 163/195 |
| gold padded 0.25 / 0.5 / 1.0 s each side | 0.830 / 0.718 / 0.573 |
| gold followed immediately by a backchannel word / including one | 28/195 / 6/195 |
| smoke-run under-merges (gold vs predicted) | 40.68-44.98 vs 40.44-42.72; 28.52-42.72 vs 40.44-42.72 |

**Expected gain.** Idea 23 says +0.02 to +0.04 score. Idea 57 says three to six points, which Fable calls optimistic on that many spans.

**Decisive judge note.** Opus on 23.

> The bench output shows the real failure is UNDER-merging, and span_from_ids' 'extend while ids are within 2' rule is a merge policy nobody ever fitted, so scoring the window is the right fix.

Fable on 57.

> +3 to +6 is optimistic on 195 spans; measure the 'right neighbourhood' rate after 55 and 46 and only fit a re-ranker if the within-candidates oracle gap is still >0.05.

**Disagreement.** Real. Opus reads the smoke run as under-merging and warns that idea 46's "one unit by default" points the wrong way at the margin; Fable ranks 46's tightening as do-tonight. Both agree on the length guard, the padding ban and the backchannel rule, and both say do not build 57, 65 and 73 all three. Both judges say the smoke run is two conversations.

**First experiment.** From tonight's full bench, log the raw `segments` list per question; build candidates from two units before the first cited id to two after the last; compute the oracle within those candidates against the LLM's raw pick. If the gap is above 0.05, fit the re-ranker; if not, the cheap version is enough. Also histogram cited-unit counts against the number of sentences the gold needs.

**Plugs into.** `model.py` `span_from_ids` (line 244 to 261). `bench/llm/prompts.py` postprocess for the `units` variant. `bench/llm/bench.py` already keeps the model's parsed JSON per question under `questions[].raw` (the smoke run has it), so `segments` and `quote` are available for the replay without a code change.

## Below the ten: cheap, both judges do-tonight, under two hours each

| item | ids | what | where |
|---|---|---|---|
| Coerce, never raise | 12, 47 | `api.py` calls `validate_response` after `predict` and lets it raise, so any of eight shape violations becomes a 500 and ten marks. Add a `coerce()` that pads or truncates the three lists, nulls half-filled or reversed intervals, clamps to the audio duration, converts numpy types, and logs at ERROR. Sanitize transcript text before the prompt and any log; sample_81 carries a lone UTF-16 surrogate that crashes a cp1252 print. | `example.py` `predict`, before the DTO; `model.py` `render_transcript` |
| Hold-out and LOCO | 52, 38, 62 | Freeze ten conversations by `transcript_id` never used for prompt or threshold decisions; a `--loco` flag in the bench that refits every scalar with one conversation held out and prints that number first. Question-level splits leak because the near-duplicate pair siblings land on both sides. | `bench/llm/bench.py`; a `bench/holdout.txt` |
| Write the number down | 26 | The oracle within two sentences of the retriever's pick versus the unrestricted oracle, so nobody adds retrieval pre-filtering at two in the morning. | `bench/README.md` |
| Breakage detectors, not priors | 76, 58 | Yes count per conversation outside the observed band, or a span inside the greeting or the goodbye, means something broke. Log it; retry only the questions where two prompt variants disagree. Never force a count. | `model.py` `answer_all`, after the results |
| Skip lists | 53, 77 | Adopted as the traps table below. | this page |

| quantity behind these | value |
|---|---|
| shape violations `validate_response` raises on | 8 |
| near-duplicate pair members among the training questions | 94/390 |
| oracle within two sentences of the lexical argmax / unrestricted | 0.643 / 0.824 |
| yes per conversation, observed band | 3 to 7 |
| earliest gold start / latest gold end before the audio end | 4.8 s / 5.8 s |
| digit-bearing questions / hard negatives with a digit | 28/390 / 10/142 |

## Tonight on the H100

Preconditions, all on the laptop and the login node, from `bench/hpc/README.md`: DTU VPN up, `bash bench/hpc/sync.sh up`, `bash bench/hpc/env.sh all`, `bash bench/hpc/env.sh check` showing the venvs and the ASR weights. Items 1, 2 and 4 are GPU jobs. Items 3 and 5 are CPU work that runs on the laptop while the jobs are queued. Each item names the code change it needs first; none of them exists in the repo tonight.

### 1. ASR sweep with the hygiene variant (ideas 4 and 6)

Code change, about thirty minutes. `bench/asr/run_faster_whisper.py` gets `--temperature` (a single float; default keeps the ladder) and `--no-condition` (sets `condition_on_previous_text=False`) and `--hallucination-silence` (float), with the tag suffix `+clean` when the first two are set together, following the existing `+vad` / `+beam<N>` rule. `bench/hpc/asr_bench.lsf` gets one more step line after `fw-large-v3`:

```
step fw-large-v3-clean  venv-asr  bench/asr/run_faster_whisper.py --model large-v3 --temperature 0 --no-condition --hallucination-silence 2 "${common_args[@]}"
```

Submit on the cluster:

```
cd /dtu/blackhole/1e/205502/nordic/medical-appointment && bsub < bench/hpc/asr_bench.lsf
```

If the queue is busy, a shorter first pass covers the fingerprint candidates and the hygiene variant:

```
STEPS="fw-large-v3 fw-large-v3-clean whisperx hf-whisper parakeet-v2 mms-large-v3 qwen3-align-lv3 compare" bsub < bench/hpc/asr_bench.lsf
```

Read `bench/results/asr/summary.md`. Table A gives the start and end medians per tag; table B gives the median distance to the nearest word boundary. Add a residual MAE per edge per tag to `compare.py` if it is missing. Decision rule, in the table.

| observation | decision |
|---|---|
| a tag with start and end medians inside +-0.05 s and boundary distance at or under 50 ms | that tool is the generator; serve its timestamps |
| no tag under 0.09 s residual MAE | keep large-v3+clean with refit constants |
| large-v3+clean per-file `seconds` max well under the current 39.1 s | adopt the decode settings in `model.py` |
| large-v3+clean word count differs from large-v3 by more than a few words on any file | inspect that file before adopting |

Then, on the laptop after `sync.sh down`, `python span_ceiling.py --model large-v3+clean` for the refit (idea 27).

### 2. Full answering bench with P(yes) recorded (ideas 8, 2, 10)

Code change, about thirty minutes. `bench/llm/bench.py` `Client.body` adds `'logprobs': True, 'top_logprobs': 5`; `run_question` records `p_yes` from the top logprobs at the answer token (with `answer` moved first in the schema for this run), and confirms the raw `quote` and `segments` are kept per question. Submit with the three models report 04 ranks first, on the large-v3 transcripts already synced:

```
MODELS="Qwen/Qwen3.8-27B Qwen/Qwen3.6-27B openai/gpt-oss-20b" ASR_TAGS="large-v3" VARIANTS="units units-claim words" bsub < bench/hpc/llm_bench.lsf
```

Once job 1 has written the other tags, resubmit with `ASR_TAGS="large-v3 large-v3+clean parakeet-tdt-0.6b-v2 whisperx-large-v3"`; existing result files are skipped.

Read the `summary` block of each result: `accuracy_by_type` rows for positive and hard_negative side by side, `mean_tiou`, `tiou_answered_yes`, `missing_spans`, `yes_rate`, `conversation_wall_s.worst`. This replaces the smoke number. Decision rule, in the table.

| observation | decision |
|---|---|
| hard_negative row below the lexical rule's row | the model is worse than a regex on that row; try the next model or the `units-claim` variant |
| `tiou_answered_yes` far below the single-sentence oracle | the selector cites the wrong sentence often; idea 10's right-neighbourhood check comes before any extent work |
| positive row well below one | idea 1 has headroom; run its validation pair as soon as the endpoint is up |
| `p_yes` reliability diagram flat or near-binary | idea 8 needs Platt scaling or is not implementable with this model |

| reference rows for the decision | value |
|---|---|
| lexical coverage rule, accuracy | 0.744 (positive 129/195, hard_negative 109/142, off_topic 52/53) |
| single-sentence oracle with offsets | 0.694 |

### 3. Quote-anchoring oracle and replay (idea 2), CPU

New script `bench/span_quote.py`, on the laptop, over `transcripts/*.large-v3.json` and the CSV. For each annotated span, take the words the gold span covers as the stand-in quote, align it back with difflib after the same normalisation the serving code will use, apply the start rule from the first word's end and the end constant, clamp to the containing sentence, score. Then drop and add one and two words at each end and re-score. Then, when job 2 lands, read each question's returned `quote` from the result file and score the quote-anchored span against the unit-id span on the same question.

| expected number | value |
|---|---|
| perfect stand-in quote, unclamped word run | about 0.91 |
| perfect stand-in quote, inside the correct sentence | about 0.81 |
| whole correct sentence | about 0.69 |
| ship threshold on real model quotes | mean tIoU at least 0.02 above the unit-id span and no rise in the under-0.5 count |
| fallback trigger | fewer than three matched words, or two match sites |

### 4. Clause units, ceiling and bench (idea 7), GPU after item 2

Code change, about one hour. `model.py` `make_units` splits also at `, ; :` when `CLAUSE_SPLIT=1`, and `render_transcript` prints sub-ids as `[12a]`, `[12b]` under the sentence index so the id space the model sees stays sentence-sized. `span_ceiling.py` must accept the same flag and `--k 6`. On the laptop first:

```
CLAUSE_SPLIT=1 python span_ceiling.py --model large-v3 --k 6
```

Expect the clause ceiling row. Then one more bench run for the best model from item 2, on the cluster:

```
CLAUSE_SPLIT=1 MODELS="<best from item 2>" ASR_TAGS="large-v3" VARIANTS="units" bsub < bench/hpc/llm_bench.lsf
```

The result file needs a distinct name; pass `--suffix .clause` through the job or rename the output. Compare `mean_tiou` and the under-0.5 count against item 2's `units` run. Decision rule: ship if mean tIoU rises and the positive row does not fall.

### 5. Lexical rung and the ladder replay (idea 5), CPU and laptop GPU

Implement `fallback_answer(units, question)` in `model.py` (tokenize, stem, stoplist, per-transcript IDF, windows of one to three units with length normalisation, the fitted offsets, a coverage threshold for yes). Score it alone against the CSV on the cached transcripts. Then, against the local Ollama endpoint:

```
LLM_TIMEOUT=0.5 python local_evaluator.py
```

Every question fails onto the rung. Then with a `FAKE_ASR_DELAY` knob that pushes five of thirty-nine conversations over budget, once the slot and watchdog exist.

| expected number | value |
|---|---|
| rung alone, always yes | about 0.51 |
| rung alone, coverage threshold, LOCO | 0.53 to 0.585 |
| `local_evaluator` with the LLM forced to fail, failed conversations | 0 |
| `local_evaluator` with five forced ASR stalls, timeouts | 0 |
| worst round trip under the stall | under 52 s |

## Traps

Ideas at least one judge marked `trap`, or that a measurement rejects. Do not re-propose without a new number.

| idea | rejection | number | verdicts (Opus, Fable) |
|---|---|---|---|
| 9 Prune candidates with the lexical scorer before the LLM sees them | Restricting the answerer to the retriever's neighbourhood caps the oracle far below the unrestricted one; the prompt is small anyway | oracle 0.643 vs 0.824; median prompt about 460 tokens | trap, trap |
| 35 Retrieve first, verify against the passages alone | Retrieval recall caps positive recall before the model reads a word. The quote-verification half is idea 2. | recall@3 0.81 | maybe, trap |
| 8 Warm the prefix cache while ASR runs | Removes a cost that does not exist on a serving GPU, depends on chunked ASR (idea 5) and on byte-identical prefixes that Whisper's revisions break | about 700 prefix tokens, tens of ms | trap, trap |
| 25 One LLM call for all ten questions (and 32, batch visible to the model) | A shifted list costs ten marks; the LLM half is a few seconds of sixty, so there is no latency to buy. 32's cross-question context is testable but only ships if the hard_negative row moves. | LLM wall 6.3 s per conversation in the smoke run | trap, trap (25); maybe, maybe (32) |
| 63 Synthetic consultations with gold by construction | Synthetic gold lacks the annotator-tool bias that is the whole offset story; six-plus hours | gold on a 0.02 s grid, offsets +0.36 / +0.12 | trap, trap |
| 5 Chunk the audio at silences and batch the ASR | Perturbs the timestamps the ceiling and the fitted offsets rest on; the RTF spikes are fallback loops that idea 6 removes | RTF p50 0.092, tail files 43 / 57 / 81 | trap, maybe |
| 11 Keep the GPU warm between requests | The RTF tail is decode fallback and cold start, not thermal; a dummy job can collide with a real request | sample_81 and 57 have 78 and 77 words under p 0.35 | maybe, trap |
| 24 Lexical floor gate on off-topic; 40 zero-overlap gate; 59c and 72's off-topic fast path | Positives with zero lexical overlap exist and each costs twice; the answerer already gets off-topic | 11/195 positives at zero coverage; smoke run off_topic 3/3 | maybe, trap (24); maybe, maybe (40) |
| 33 Assign one passage per yes across the batch | Positives legitimately share spans, so the constraint is half absent; a hard negative citing the same sentence answers no anyway | 47/195 positives overlap another positive's span | maybe, trap |
| 4 Attempt-level slack bank | Nothing proven to spend the bank on; self-consistency needs a positive temperature and a probability that does not exist yet; order-dependent runs | bank about 1,500 s per attempt at a 20 s pipeline | maybe, trap |
| 3 Compute plan from the MP3 header | Solves a scarcity that does not exist on a serving GPU | about 40 s of slack per clip | maybe, maybe |
| 36 Ask each claim in both polarities | No measurement at all; automatic negation of tag questions is noise | 0 measurements | maybe, maybe |
| 22 Fuse Silero VAD with the Whisper edge | Equal MAE, unmeasured correlation; gold edges sit inside speech, so VAD measures a different thing. Only the twenty-minute correlation check is worth doing. | start MAE 0.157 vs 0.153 | maybe, maybe |
| 13 Per-span forced alignment on crops | Subsumed by idea 4's whole-transcript MMS run; edges are not where the loss is | spans on the right sentence already score 0.96 to 0.99 | maybe, maybe |
| 2 Front door and worker as separate processes | A supervisor with restart plus the catch-all in `predict` buys most of it for one hour | 5 h | do-this-week, maybe |
| 6 Hedged ASR, small and large model together | Contention on the 8 GB card; the tail is gone after idea 6 | 2 models resident | maybe, maybe |
| 7 One prefill, ten decodes | A property of vLLM V1, not a project | prefix caching on by default | maybe, do-this-week |
| 39 Attempt-wide yes-rate controller | Hurts a calibrated answerer; makes runs order-dependent; gate on a validation yes rate outside 0.45 to 0.55 | -0.006 at zero bias | maybe, maybe |
| 56 De-inflate the first word per token | Inside measurement noise; take idea 68's one-parameter rule instead | ceiling 0.816 to 0.824 | maybe, do-this-week |
| 17 Two-branch edge offsets | Self-declared null at sentence level; bucket sizes too small | 0.690 vs 0.695; n = 9 and 20 | maybe, maybe |

Rules the data rejects, from ideas 53 and 77 and the measurements, each with its number.

| rule | number |
|---|---|
| Pad the span to be safe | gold padded 0.25 / 0.5 / 1.0 s scores 0.830 / 0.718 / 0.573; scaling the served span by 1.15 scores 0.474 vs 0.498 |
| Merge the next unit to be safe | retriever single unit 0.498, always-merge-next 0.342 |
| Force five yes per request | yes per conversation runs 3 to 7; forcing five caps accuracy at 0.877 |
| Snap edges to audio energy or VAD | energy onset MAE 0.298 vs 0.117; snapping both edges drops the clause ceiling 0.875 to 0.786 |
| Shift the citation to the following turn | gold sentence minus lexical argmax is 0 in 128/195, +1 in 6; shifting scores 0.031 |
| Bigram bonus in the lexical scorer | 0.482 vs 0.501 |
| Dense retrieval as the cheap fallback | MiniLM 0.400 vs lexical 0.521, 400x the compute |
| A numeric verifier or audio-LLM crop check | 28/390 questions carry a digit, 10 of 142 hard negatives; digit tokens mean p 0.943 |
| Diarization or audio-LLM localisation | best open audio model 31 percent mean IoU on TAG-Bench; spans are single utterances |
| A 27B model on the 8 GB laptop | CPU offload turns seconds into a minute |
| Any filename-keyed cache in production | 56 unused ids in 1..95; hidden files reuse the naming |
| Fine-tune on the training rows without conversation-level CV | 39 conversations, 94/390 questions in near-duplicate pairs |
| Tune toward the hard_negative row alone | a recovered hard negative is worth 0.10 points, a lost positive 0.26 to 0.36 |
| Key on question order or slot | positives' request order vs time order 214 concordant / 198 discordant; yes rate by slot within noise |
| Key on the question id | not in the request DTO |

## Measurements that changed a decision

Each row is one number that moved an idea into the ten, out of it, or changed how it is built.

| measurement | number | decision it changed |
|---|---|---|
| The ported scorer never reads the boolean | `record(label=1, prediction=0, same span)` gives tIoU 1.000 | idea 1 exists; idea 8 becomes conditional on the A/B |
| Score granularity per flipped answer | 0.0021 | validation attempts are a measurement instrument (idea 3) |
| Validation score SD from sampling | 0.027 on 19 conversations | pair every validation comparison; cap unpaired runs (3, 62) |
| Word-run ceiling vs sentence ceiling | 0.927 to 0.931 vs 0.824 | quote anchoring is the largest tIoU lever (2) |
| Unclamped growth across a pause | 0.935 to 0.650 | the clamp and fallback rules are mandatory, not optional (2) |
| Asymmetric trimming cost | one word long 0.747, one word short 0.564 | resolve ties outward inside the clamp (2, 19) |
| Inside the correct sentence, whole vs sub-clause | 0.690 vs 0.806 | sub-clause trimming is the binding gap, not edges (2) |
| Clause ceiling | 0.882 vs 0.824, under-0.5 spans 11 vs 23 | clause units stay in the ten (7) |
| Retriever neighbourhood oracle | 0.643 vs 0.824 | no retrieval pre-filtering (traps 9, 35; note 26) |
| Retriever recall@3 | 0.81 | retrieve-then-verify rejected; verifier half kept (35) |
| Lexical fallback vs the except branch | 0.512 vs 0.200 | the lexical rung replaces `(True, None)` (5) |
| MiniLM vs lexical | 0.400 vs 0.521, 198 ms vs 0.19 ms | lexical, not dense, for the rung (5, 72) |
| No global deadline in `model.py` | 39 s + 25 s = 64 s | the ladder is structural, not optional (5) |
| Bootstrap timeout probability, laptop profile | P(at least one) 0.37, P(five in a row) 0.000 | the risk is the silent ten marks, not the abort; Fable reads it as a laptop number (5) |
| The slow ASR files are the fallback files | samples 43, 57, 81; RTF max 0.397 | decode hygiene (6) and rejects thermal (11) and chunking (5) |
| Energy onset vs corrected Whisper edge | MAE 0.298 vs 0.117 | the waveform-edge family is dead (4, 69, 22) |
| Start rule from the first word's end | MAE 0.089 vs 0.117 | idea 68's rule over 17's buckets (2, 4) |
| Gold grid | 195/195 on 0.02 s; 15/195 on 0.08 s | fingerprint the tool; Parakeet's grid is a cost (4, 27) |
| Padding cost | 0.830 / 0.718 / 0.573 | no padding anywhere (traps) |
| Yes per conversation | {3:7, 4:10, 5:7, 6:6, 7:9} | no quota; the band is a breakage detector only (76, 58) |
| Per-question weight ratio | exactly 3 | the threshold argument (8), gated on idea 1 |
| Monte Carlo threshold gain | +0.018 / +0.029 / +0.007 at 82 / 75 / 88 percent | the gain is small and shrinks as the answerer improves (8) |
| Positives with zero lexical coverage | 11/195 | no hard lexical gate (traps 24, 40, 59c) |
| Smoke run off_topic and hard_negative rows | 3/3 and 4/4 | no headroom for off-topic gates, with the n=7 caveat |
| Hard negatives are absent claims, not twins | 77/142 under half coverage; 5/142 with a lexical sibling | prompt for "not established" (61); pair trick down-ranked (31) |
| Near-duplicate pairs | 51 pairs, 40 yes/no, 11 yes/yes, 0 no/no | forced choice stays maybe; the yes/yes pairs forbid a hard constraint (31) |
| Tag questions | 20/21 yes; the one no is a real row, sample_17 | soft prior only, after an A/B (34, 60, 70) |
| Digit-bearing questions | 28/390; 10/142 hard negatives | no numeric verifier, no audio LLM (77) |
| Body size vs proxy default | 4.95 MB vs 1 MB | pre-flight is mandatory (9) |
| Unused ids in the training range | 56 of 1..95 | no filename cache; validation and evaluation are disjoint (9, 49) |
| Positives sharing a span | 47/195 | no assignment solver, no diversity penalty (33) |
| Gold position band | none before 4.8 s or within 5.8 s of the end | free span sanity check (58) |
| LLM cost per conversation | 906 prompt tokens, 6.3 s wall in the smoke run | every prefill and batching idea loses its rationale (25, 32, 7, 8, 3, 4) |
| Lone surrogate in a transcript | U+D391 in sample_81 | text sanitizer (47) |
| The bench on disk | 2 conversations | "realised 0.49" is not a number; the full bench is tonight's item 2 |

## Where each of the ten plugs in

| idea | `model.py` | bench |
|---|---|---|
| 1 span on every question | `answer_all` / `one()`: compute the span regardless of `yes`; `SPAN_ON_NO` env | `bench.py` scores both policies from one run |
| 2 quote anchoring | new `span_from_quote`, called from `one()`, `span_from_ids` as fallback; number and unit normaliser shared with the prompt | `prompts.py` `quote-anchor` variant; `span_ceiling.py` word-run mode; new `span_quote.py` oracle |
| 3 validation A/B | policy toggles as env vars read at import; request dump in `example.py` | `results/validation_log.md` |
| 4 aligner fingerprint | `START_OFFSET` / `END_OFFSET` from `offsets.json` keyed by `ASR_MODEL` | `asr_bench.lsf`, `compare.py` residual MAE per tag, `span_ceiling.py --fit` |
| 5 ladder and lexical rung | `answer_all` slot and watchdog; `one()` except branch; `fallback_answer` | `lexical` variant in `prompts.py`; `local_evaluator` runs with `LLM_TIMEOUT=0.5` and `FAKE_ASR_DELAY` |
| 6 decode hygiene | `transcribe()` kwargs | `run_faster_whisper.py` flags and `+clean` tag; `transcribe_cache.py` the same |
| 7 clause units | `make_units`, `_TERMINAL`, `render_transcript` sub-ids, `CLAUSE_SPLIT` env | `span_ceiling.py --k 6`; bench run with `--suffix .clause` |
| 8 P(yes) and threshold | `SCHEMA` answer first; `ask_llm` reads logprobs on vLLM; `YES_THRESHOLD` in `one()` | `bench.py` body and record `p_yes`; LOCO sweep script |
| 9 pre-flight | none | new `bench/preflight.py`; VM proxy and supervisor config |
| 10 extent selection | `span_from_ids` adjacency and soft length guard; optional re-ranker over candidates | `bench.py` keeps raw `segments`; right-neighbourhood script over the result file |

## Attribution and what was already ours

Already in `00-synthesis.md`: the aligner fingerprint (idea 4, experiment 1), clause units (idea 7, experiment 2), the quote field in the schema (half of idea 2), the cloud VM with a raised body limit (half of idea 9), the padding cost, the offset constants and the no-diarization, no-audio-LLM verdicts.

New from the brainstorm: spans on every question (1), quote-derived edges with the clamp (the other half of 2), the paired validation design and the request dump (3), the refusal of waveform edges and the offsets-as-config-lock (inside 4), the deadline ladder and the lexical rung (5), decode hygiene (6), the probability and threshold work (8), the outside-the-LAN checklist (9), the unfitted gap rule in `span_from_ids` and the candidate scoring (10), the repair layer, the hold-out discipline, and the traps table.

Lenses that carried the ten: `fable-failure` four, `opus-answering` two, `fable-data` two, `fable-wildcard` one, `opus-localization` one. Every one of the ten had at least one duplicate on the other model family, which is the strongest evidence the pool has that these are the real levers.
