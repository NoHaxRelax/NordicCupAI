# Findings log

Dated measurements on our own data. One entry per finding, newest at the bottom. Every number here comes from a script in the repo that can be re-run; the script is named in each entry. Opinions go in the other research files, not here.

Terms: tIoU is temporal intersection over union, the overlap of two spans divided by their union. ASR is automatic speech recognition. LOCO is leave-one-conversation-out cross-validation.

## 2026-09-17

### 1. The annotated timestamps sit on a 20 ms grid

All 390 annotated `evidence_start` and `evidence_end` values in `data/question_train.csv` are multiples of 0.02 s. Three carry float noise (for example 28.520000000000003). Only 15 of 195 spans have both edges on an 80 ms grid, which is what a Parakeet or Qwen aligner would produce. Source: committee report 03 and `bench/asr/compare.py`.

### 2. Gold spans are one utterance long

| statistic of span length, n = 195 | seconds |
|---|---|
| minimum | 0.16 |
| p10 | 1.32 |
| p25 | 1.94 |
| median | 2.88 |
| p75 | 4.04 |
| p90 | 5.44 |
| maximum | 14.20 |

Ten spans are shorter than 1.3 s and cover a phrase inside a sentence. Being 0.5 s off at both edges gives a mean tIoU of 0.66 over these lengths; 1.0 s off gives 0.45. Script: the statistics block in the session notes, reproducible with `span_ceiling.py`.

### 3. Whisper large-v3 word starts are early and word ends are early, by different amounts

Oracle-selected sentence merges (up to four sentences) against the gold spans, faster-whisper large-v3, beam 5, no VAD, 39 conversations. Script: `span_ceiling.py --model large-v3`.

| quantity | value |
|---|---|
| ceiling, Whisper segments as units | 0.556 |
| ceiling, sentence units | 0.644 |
| ceiling, merges of up to four sentences | 0.753 |
| gold start minus merge start, median (p25, p75) | +0.36 s (+0.24, +0.48) |
| gold end minus merge end, median (p25, p75) | +0.12 s (+0.06, +0.20) |
| ceiling after the two constant shifts | 0.824 |

### 4. Where the gold edges fall inside Whisper's words

Script: `bench/asr/compare.py`, table B and C. f is the position of the gold edge as a fraction of the nearest Whisper word, 0 at the word's start and 1 at its end.

| edge | median f (p10, p90) | inside the word | after the word |
|---|---|---|---|
| start vs first word | 0.67 (0.09, 0.92) | 95% | 2% |
| end vs last word | 1.27 (0.05, 1.71) | 16% | 75% |

Median distance from a gold edge to the nearest Whisper word boundary is 120 ms at both edges; only 7 percent of starts and 4 percent of ends land within 20 ms of one. Whisper's word intervals therefore begin too early and end too early relative to the annotation, and raw faster-whisper word timestamps cannot be the tool that produced the annotation, since that tool's boundaries would coincide with the gold edges.

### 5. The best cheap transformation of Whisper's edges

Script: `bench/asr/fit_edges.py --model large-v3`, every rule fitted LOCO, errors out of sample.

| start rule | MAE | within 100 ms |
|---|---|---|
| raw first-word start | 0.595 s | 7% |
| first-word start + 0.36 | 0.389 s | 45% |
| first-word end - 0.14 | 0.356 s | 55% |
| first-word start + 0.73 x word duration | 0.361 s | 55% |
| linear in word duration (least squares) | 0.449 s | 16% |

| end rule | MAE | within 100 ms |
|---|---|---|
| raw last-word end | 0.407 s | 19% |
| last-word end + 0.12 | 0.351 s | 64% |
| last-word end + 0.23 x following pause | 0.362 s | 62% |
| linear in word duration (least squares) | 0.470 s | 6% |

| edges applied to the oracle merge | mean tIoU |
|---|---|
| raw | 0.753 |
| start + 0.36, end + 0.12 | 0.822 |
| first-word end - 0.14, last-word end + 0.12 | 0.832 |

The mean absolute errors are large because a minority of spans are sub-sentence and the oracle merge is the wrong unit for them; the within-100-ms share is the informative column. Least-squares fits lose to median rules because those outliers dominate them. The remaining error is in Whisper's boundaries themselves, so the next gain has to come from a better aligner, not from a better formula.

### 6. Yes answers are balanced per set, not per conversation

Training set: 195 yes, 195 no. Yes per conversation: 3 in 7 conversations, 4 in 10, 5 in 7, 6 in 6, 7 in 9. There is no per-conversation quota to exploit. The set-level balance gives a label-free calibration check on a validation run: our yes count over 190 questions should land near 95.

### 7. Hard negatives are mostly absent claims, not twins of a positive

Of 142 hard negatives, 23 (16 percent) share at least a third of their content words with a positive question about the same conversation, and 5 share half. Off-topic questions share nothing with any other question. The near-miss pairs in the brief are the clean case, not the typical one.

### 8. The evaluator scores spans on every annotated yes question, whatever we answered

`local_evaluator.Statistics.record` and `utils.mean_temporal_iou` select questions by the gold label alone and never read the returned boolean. `utils.validate_response` accepts a span next to `false`. So a positive answered no still earns its tIoU if a span is attached, and a span next to a true negative is discarded. Whether the live service does the same is untested; the README prose says a missed positive scores zero on both halves. One paired validation run settles it. Found by the committee (research/06, idea 1) and verified in the code.

### 9. The audio is synthetic, clean, and never overlaps

By ear (Elias, samples 4 and 5): two clear synthetic voices, no overlap, no noise. Whisper large-v3's transcript of sample 4 needed only two punctuation changes. Consequences: diarization adds nothing, pause-based unit splitting is reliable, and the ASR comparison will be decided by timestamp behaviour rather than word accuracy.

### 10. Serving numbers on the laptop (RTX 5070, 8 GB)

| step | value |
|---|---|
| Whisper large-v3 alone, RTF | 0.095 |
| Whisper large-v3 with the LLM resident, sample 5 | 45 s (17 s alone) |
| distil-large-v3 with the LLM resident, per conversation | 2 to 5 s |
| qwen3:4b, ten questions, four Ollama slots | 5 to 15 s |
| Ollama with ten slots at 8192 context | 11.5 GB cache, 7 GB spilled to CPU, every question over 25 s |

The 8 GB card cannot hold large-v3 and a 4B LLM together without one of them slowing down. The validation run of 2026-09-17 served all 19 conversations in 7 to 18 s each on distil-large-v3 plus qwen3:4b.

### 11. What the 39 consultations are about

Extracted from the large-v3 transcripts by the local qwen3:4b model with a JSON schema (`bench/describe_topics.py`, output in `bench/results/topics.json`), so counts are approximate; conversations where it extracted nothing are counted as zero.

| quantity | value |
|---|---|
| conversations | 39 |
| distinct conditions after folding diabetes variants | 38 |
| conditions seen in exactly one conversation | 35 |
| conditions per conversation, mean / min / max | 1.3 / 0 / 8 |
| conversations flagged general practice | 32 |

Most frequent: diabetes (11), asthma (3), thyroid dysfunction (2), constipation (1), gastroesophageal reflux disease (1), menopausal symptoms (1), cervical disc prolapse (1), chronic pain (1). The set is broad general practice with a diabetes and asthma head and a long tail of one-off conditions, so there is no single illness to specialise on, and any vocabulary list for ASR biasing has to be general medical, not disease-specific.

### 12. Local baseline of the served pipeline, all 39 training conversations

Live endpoint on the laptop (distil-large-v3, qwen3:4b via Ollama, START_RULE first-word-end, ASR_CLEAN, SPAN_ON_NO), scored by `local_evaluator.py` end to end over HTTP.

| quantity | value |
|---|---|
| accuracy | 0.956 (373/390) |
| positive | 0.918 (179/195) |
| hard_negative | 1.000 (142/142) |
| off_topic | 0.981 (52/53) |
| mean tIoU over 195 annotated yes | 0.458 |
| no span returned | 0 |
| score | 0.657 |
| round trip per conversation, mean / worst | 8.2 s / 11.7 s |

The answering half is close to its ceiling on this model already; the missing points are in the spans (0.458 realised against a 0.832 oracle at sentence granularity). This is the reference number for the first deliberate validation run.

### 13. First validation scores

| attempt (2026-09-17) | served by | score |
|---|---|---|
| 13:37, queued by accident during a server restart | distil-large-v3 + qwen3:4b, unit-start rule (+0.36 / +0.12), nulls on no | 0.5869 |
| 15:45 | server had crashed 2 minutes earlier (python.exe fail-fast 0xc0000409 in ucrtbase.dll during an external request); every request 502 | 0.0000 |

The local baseline on the 39 training conversations with the newer defaults is 0.657 (entry 12). The crash led to a supervisor loop around api.py that restarts it on exit. Portal scores are read with `bench/portal_status.py` (key file gitignored).

### 14. Validation run A: 0.6086 with the new serving defaults

Queued from `bench/portal_status.py --queue` at 16:17, finished 16:21, 19 conversations, no errors. Served by distil-large-v3 + qwen3:4b with START_RULE first-word-end (-0.14), END_OFFSET +0.12, ASR_CLEAN, SPAN_ON_NO=1. Per-conversation time 7 to 13 s. Yes answers 85 of 190 (0.447; the set is balanced at 0.5), so the answerer still leans no on unseen conversations.

| run | policy | validation score |
|---|---|---|
| morning (accidental) | old offsets, nulls on no | 0.5869 |
| A, 16:17 | new offsets, spans on every question | 0.6086 |
| B | new offsets, nulls on no | pending |

A minus B, divided by 0.6, is the mean tIoU the live scorer credits on spans returned with a no answer. Local estimate of the same difference on three conversations: 0.014 in score.

### 15. Validation A/B: the live scorer does not credit spans on no answers

Runs A (16:17) and B (16:23) used byte-identical code and models; the only difference was SPAN_ON_NO (A returns the located span for every question, B returns null on no).

| run | policy | score |
|---|---|---|
| A | spans on every question | 0.6086 |
| B | nulls on no | 0.6062 |
| difference | | 0.0024 |

If the live scorer credited spans returned with a no answer, A minus B should have been roughly 0.6 x (share of positives answered no, about 0.15 at a 0.447 yes-rate) x (their span tIoU, about 0.4), around 0.03. The observed 0.0024 is within run-to-run noise of a temperature-0 Ollama model with four parallel slots. Conclusion: the README prose holds on the service (a missed positive scores zero on both halves), and `local_evaluator.py` overstates our score by crediting those spans. Consequences: keep SPAN_ON_NO=1 (harmless), read local tIoU from the "nulls on no" policy in `bench/llm/bench.py`, and the yes-threshold idea (research/06, idea 8) is back on the table: at a 0.447 yes-rate every recovered positive is worth its accuracy point plus its whole tIoU.

Noise floor, measured afterwards: the same ten training conversations scored twice with identical settings (qwen3:4b, temperature 0, four Ollama slots) gave identical answers but different spans on some questions, score 0.648 vs 0.654 under spans-on-no and 0.637 vs 0.643 under nulls-on-no. So two identical runs differ by about 0.006 on ten conversations, and the validation A/B gap of 0.0024 is inside that. The local spans-on-no benefit (0.011 on these ten) sits above it, which is why it shows locally and not on the service.

### 16. Validation leaderboard snapshot, 2026-09-17 16:35

25 of 44 teams have a medical-appointment validation score. Top eight sit between 0.720 and 0.746; then 0.661, 0.623, us at 0.609 in 11th, 0.607, and a long tail at or below the 0.454 mark, seven teams at the 0.200 all-yes baseline.

| rank | team | best validation |
|---|---|---|
| 1 | Eirik Solberg (NO) | 0.7457 |
| 2 | execve (DK) | 0.7447 |
| 3 | Cybotrix (IS) | 0.7390 |
| 8 | Iceland Here We Go (DK) | 0.7200 |
| 9 | Team Ambolt Interns - N (DK) | 0.6606 |
| 11 | Powered by Smørrebrød (us) | 0.6086 |

Reading: with accuracy near 0.9 (worth 0.36 of the score), our validation tIoU is about 0.41. A 0.745 score at similar accuracy needs a mean tIoU near 0.62. The gap to the top is spans, not answers, which is what the ASR sweep and the quote-anchoring work target.

### 17. Whisper large-v3-turbo's word ends coincide with the annotated ends

faster-whisper large-v3-turbo, beam 5, no VAD, 39 conversations on the laptop (RTF 0.043, 2.5x faster than large-v3). Scripts: `bench/asr/compare.py --tags large-v3,large-v3-turbo`, `bench/asr/fit_edges.py --model large-v3-turbo`.

| quantity | large-v3 | large-v3-turbo |
|---|---|---|
| oracle ceiling, merges of up to four sentences, raw edges | 0.753 | 0.812 |
| ceiling with fitted constant shifts | 0.824 | 0.847 |
| ceiling with the best rule pair | 0.834 (first-word end -0.14, end +0.12) | 0.857 (start + 0.56 x first-word duration, end -0.02) |
| gold start minus model start, median | +0.36 s | +0.28 s |
| gold end minus model end, median (p25, p75) | +0.12 (+0.06, +0.20) | -0.02 (-0.06, +0.02) |
| gold ends within 20 ms of a word end | 4% | 49% |
| gold ends inside the last word | 16% | 72% |
| WER on the 4 hand-checked references | 3.0% | 1.8% |

Half of the annotated ends land within one 20 ms frame of a turbo word end, against 4 percent for large-v3. That is a fingerprint: whatever produced the annotations places its ends where turbo's decoder does. Starts are still 0.28 s late relative to turbo's word starts, so the start side of the annotation comes from a different mechanism (or a later shift) and still needs the fitted rule. Serving implication: switch ASR to large-v3-turbo with START_RULE first-word-end and START_OFFSET -0.20, END_OFFSET -0.02 (both from the LOCO fit), pending the WhisperX and MMS comparison on the same files.

### 18. MMS re-timing, the hygiene variant, and the hybrid: turbo alone wins

All on the laptop, 39 conversations, oracle-selection ceilings with merges of up to four sentences. Scripts: `bench/asr/align_mms.py --tag large-v3-turbo`, `bench/asr/run_faster_whisper.py --model large-v3 --temperature 0 --no-condition`, `bench/asr/compare.py`, `bench/asr/fit_edges.py`.

| tag | RTF | raw ceiling | best fitted ceiling | gold-start offset median | gold-end offset median | starts within 20 ms | ends within 20 ms |
|---|---|---|---|---|---|---|---|
| large-v3 | 0.111 | 0.753 | 0.834 | +0.36 | +0.12 | 7% | 4% |
| large-v3+clean (temperature 0, no conditioning) | 0.086 | 0.730 | 0.808 | +0.38 | +0.12 | 9% | 3% |
| large-v3-turbo | 0.043 | 0.812 | 0.857 | +0.28 | -0.02 | 11% | 49% |
| large-v3-turbo+mms (torchaudio MMS_FA re-timing) | +0.020 | 0.832 | 0.853 | -0.06 | -0.10 | 24% | 14% |
| hybrid: MMS start -0.06, turbo end raw | | | 0.857 | | | | |

The hygiene variant transcribes faster (per-file max 32.8 s vs 39.1 s on the shared laptop card) and with lower word error (1.2 percent vs 3.0 on the checked references) but its timestamps are worse, so it is rejected for serving. MMS gives the tightest starts of any tool so far (24 percent within one frame, the annotated start sits 60 ms before the aligned onset) but its ends run 100 ms late; combining MMS starts with turbo ends reaches 0.857, exactly what turbo reaches on its own with the first-word-end rule, so the aligner step is not worth its cost. Serving decision: ASR large-v3-turbo, START_RULE first-word-end with START_OFFSET -0.20, END_OFFSET -0.02; encoded per model in `model.py`. Note for the earlier validation runs: distil-large-v3 was served with large-v3's fitted offsets, which were never measured for distil.

### 19. WhisperX does not match the annotation edges either; turbo with the fitted rule is the served ASR

WhisperX (faster-whisper large-v3 plus wav2vec2 forced alignment, digits spelled out for the aligner) on the laptop, 37 of 39 files (samples 20 and 23 failed in alignment). Scripts: `bench/asr/run_whisperx.py --allow-home-cache`, `compare.py`, `fit_edges.py`.

| tag | raw ceiling | best fitted ceiling | starts within 20 ms of a word boundary | ends within 20 ms |
|---|---|---|---|---|
| large-v3 | 0.753 | 0.834 | 7% | 4% |
| large-v3-turbo | 0.812 | 0.857 | 11% | 49% |
| large-v3-turbo+mms | 0.832 | 0.853 | 24% | 14% |
| whisperx-large-v3 | 0.821 | 0.847 | 11% | 7% |

No tool reproduces the annotated starts (the best, MMS, hits one frame on a quarter of them); only turbo's decoder reproduces the ends. The annotations were therefore not produced by any of these tools as-is; the ends are consistent with a turbo-family decoder and the starts with a separate rule or a manual pass. Served configuration from here: large-v3-turbo, START_RULE first-word-end, START_OFFSET -0.20, END_OFFSET -0.02, ASR_CLEAN off. Expected sentence-granularity ceiling 0.857; the realised gain depends on the selector, which the validation run after this switch measures.

### 20. Validation run C: 0.6599 after switching the served ASR to large-v3-turbo

Queued 17:06, finished 17:10, 19 conversations, no errors, 7 to 12 s each. Only change from run A: ASR distil-large-v3 with large-v3's offsets replaced by large-v3-turbo with its own fitted rule (first-word end -0.20, end -0.02). Yes answers 84 of 190, the same as run A, so the answering half did not move and the gain is entirely spans.

| run | served ASR | score | implied validation tIoU at ~0.89 accuracy |
|---|---|---|---|
| A, 16:17 | distil-large-v3, offsets fitted for large-v3 | 0.6086 | 0.42 |
| C, 17:06 | large-v3-turbo, fitted rule | 0.6599 | 0.51 |

Gain 0.051, above the 0.006 run-to-run noise. The answerer's yes-rate of 0.44 against a balanced 0.5 is now the largest visible loss on the accuracy side and a hidden loss on the span side (every positive answered no scores zero).

### 21. Where the realised spans lose against the ceiling (qwen3:4b, turbo transcripts, training set)

Full 390-question bench on the turbo transcripts (`bench/llm/bench.py --asr large-v3-turbo`), three prompt variants, nulls-on-no policy:

| variant | accuracy | mean tIoU | score |
|---|---|---|---|
| units (cite unit ids) | 0.969 | 0.481 | 0.677 |
| units-claim (tag questions rewritten as claims) | 0.972 | 0.453 | 0.661 |
| words (cite first and last word) | 0.956 | 0.488 | 0.675 |

Against a 0.857 oracle ceiling the realised 0.48 is a selector problem, not an edge problem. Of the 185 positives answered yes under `units`: 39 percent have both edges within 0.5 s (tIoU 0.94), 27 percent point at a unit with no overlap at all, 22 percent are too short with the gold continuing left or right equally often, 11 percent spill over. The model cited exactly one unit in 184 of 186 cases although a quarter of the gold spans cover two sentences. Locating the model's verbatim quote in the transcript gives a better unit than its cited id (0.528 vs 0.505 mean tIoU) but does not rescue the wrong-place cases (8 of 51), which are the model choosing the wrong sentence outright. Changes made: the span anchors on the quote's unit with cited ids kept when adjacent (`model.anchor_ids`), and the prompt asks for every consecutive utterance the fact rests on. A bigger answering model is the remaining lever for the wrong-place quarter.

### 22. Quote anchoring plus multi-unit citations: local tIoU 0.481 to 0.524

Same bench as entry 21 (`units` variant, turbo transcripts, qwen3:4b), after `model.anchor_ids` (span built from the unit where the verbatim quote lands, cited ids kept when within two units of it) and the prompt asking for every consecutive utterance the fact rests on.

| | before | after |
|---|---|---|
| accuracy | 0.969 | 0.964 |
| mean tIoU, nulls on no | 0.481 | 0.524 |
| tIoU when answered yes | 0.505 | 0.556 |
| score, nulls on no | 0.677 | 0.700 |

Gain 0.023 in score, above the 0.006 noise floor. Deployed to the laptop endpoint for validation run D.

### 23. Extractive QA as a span locator: 0.450, below the small LLM's citations

Nikolaj's suggestion: `deepset/roberta-base-squad2` given the question and the full turbo transcript, its predicted answer characters mapped to the containing unit(s), served edge rule applied (`bench/llm/qa_locate.py`). Over the 195 annotated yes questions: mean tIoU 0.450, 108 below 0.5, against 0.556 for qwen3:4b's cited unit on the positives it answered yes and 0.857 for the oracle unit. The squad2 model points at a few answer tokens and often at the topic mention rather than the utterance that establishes the fact, so it does not replace the LLM as a selector. It runs in milliseconds, so it stays a candidate for a tie-break when the LLM's quote and cited id disagree.

### 24. Validation run D: 0.6632, the anchoring gain does not transfer

Queued 17:24, finished 17:29, 19 conversations, no errors. Only change from run C: quote anchoring plus the multi-unit citation prompt (entry 22, +0.023 locally). Score 0.6632 vs 0.6599, a difference of 0.003, inside the 0.006 run-to-run noise. Yes answers 86 of 190. Kept deployed (no harm measured), but not counted as a gain. Runs so far: 0.5869 (old offsets, distil), 0.6086 (A), 0.6062 (B), 0.6599 (C, turbo), 0.6632 (D).

### 25. Yes-threshold sweep with qwen3:4b: no gain, the probabilities are near-binary

The bench now records P(yes) at the answer token from the server's log-probabilities (`bench/llm/bench.py`, `p_yes_from_logprobs`). On the 390 training questions with the turbo transcripts (`bench/llm/threshold_sweep.py`): the score is flat from threshold 0.05 to 0.65 (0.702 to 0.706), the leave-one-conversation-out choice of threshold (median 0.15) scores 0.699 out of sample against 0.702 as answered. The model's probabilities sit at 0.0 or above 0.9 on nearly every question, so there is nothing to re-threshold. The idea stays for the larger models, whose probabilities are expected to be softer; the plumbing is in place.

### 26. Validation runs E and F: 0.6649 and 0.6759 on unchanged code

Both runs served the same configuration as run D (turbo, fitted rule, quote anchoring, qwen3:4b). E at 17:38 scored 0.6649, F at 17:47 scored 0.6759; D was 0.6632. The spread of 0.013 across three identical runs is twice the 0.006 measured on ten training conversations, so validation-to-validation noise is about 0.01 and single-run differences below that are not evidence. F also dumped the 19 validation conversations (audio, questions, our answers) to `request_dump/` for hand labelling in the oracle's `/label` page.

### 27. The validation binaries are all recoverable from the turbo transcripts: 190 of 190 in one run

Hand answers for the 19 validation conversations (agent reading `request_dump/transcripts/*.large-v3-turbo.json`, written with evidence segments and estimated spans in `bench/mine/agent_answers.md`) served from the table endpoint with null spans scored exactly 0.4000, i.e. 190 correct of 190, in a 20-second run (2026-09-17 18:13). So the validation set has 95 positives, and the nine questions where the served pipeline (run F) disagreed with the hand answers are all pipeline errors: eight positives answered no (sample 8 "Was Brentan also tried earlier?" and "tendency to develop oral thrush", 14 "redness of the conjunctiva", 24 "came to have a skin change under the breast removed", 26 "lesion already removed before this contact", 60 "diabetes well controlled", 62 "came in without any complaints", 80 "showed up in person") plus sample 3 "Has InfluvacTetra already been given?" (given at the visit). Run F's accuracy was therefore 181/190 = 0.953, which puts its mean tIoU at (0.6759 - 0.4 x 0.953) / 0.6 = 0.49, matching the local bench. The hand answers double as a held-out label set for the served pipeline (`bench/mine/agent_labels/*.json`); the pipeline is never tuned on them.

Count-query arithmetic used: with null spans score = 0.4 x correct / 190, so one question is 0.0021 and the four-decimal score gives the count exactly (`bench/mine/count_run.py`). Had the first run come back short, flipping a whole conversation changes the count by 10 - 2 x (correct in it), one run per conversation.

### 28. All 95 validation evidence spans recovered; the served pipeline's held-out diagnosis matches the training picture

`bench/mine/span_probe.py` recovered every gold span of the validation set in 358 validation runs (2026-09-17 18:20-19:20, about 9 s each): per positive question one wide probe measures the gold length exactly (IoU = L / width, the portal returns the score at full float precision), one half-open probe ending inside the gold gives the start (IoU = (m - g) / (h - A)), and one verify probe returns IoU 1.0. Five seeds sat in the wrong sentence and were located by bisection over the whole audio. The full table (hand binaries plus recovered spans) then scored exactly 1.0 on the portal. The spans are in `bench/mine/span_state.json` (`answers_md.py --spans --gold` builds the table) and on the timeline page (`bench/timeline.py`).

What the gold spans look like on validation: the hand seeds (one turbo sentence unit) overlap the gold in 87 of 95 cases but match it (IoU >= 0.9) in only 26; the annotators often include the question that prompted the answer or the confirming reply (sample 3 "no acute signs": 40.14-49.36 over three units; sample 44 "no treatment": 60.62-65.34 including the patient's "And in terms of the next step?"). Sample 80 "showed up in person" is anchored on "Please have a seat" (8.48-11.80).

Held-out diagnosis of the served pipeline (`bench/mine/val_diag.py`, spans from the request dump of the last real validation run): binaries 181/190, mean tIoU 0.463 with nulls on no, reconstructed score 0.659 (the portal said 0.6759 for run F; the dump may hold the run before it, and the noise floor is 0.01). Span outcomes on the 95 positives: exact 27, edge errors 41, wrong place 18, missed 9. Edges are unbiased (start median +0.05 s, end -0.02 s). Same conclusion as entry 21 on the training set: the answering model's sentence selection (27 of 95 scoring zero) is the loss, not the timestamp rule. The timeline page (bench/ref/timeline.html, published as an artifact) shows the training breakdown too: exact 64, too little 31, too much 28, shifted 25, wrong place 35, missed 12 of 195.

Rule kept: the validation labels are a held-out measurement only; the served pipeline is never tuned on them.

### 29. Cluster ASR sweep: large-v3-turbo keeps the best timestamp ceiling; the aligner and CTC models do not beat it

Job 29428666 (A10, 68 min, 2026-09-17 18:20-19:29) transcribed the 39 training conversations with twelve configurations and ran `bench/asr/compare.py` (full table in `bench/results/asr/summary.cluster.md`). Oracle sentence-merge ceiling with per-model fitted edge offsets (in-sample fit, so slightly optimistic for every row alike):

| tag | ceiling | RTF on A10 | start offset | end offset |
|---|---:|---:|---:|---:|
| faster-whisper large-v3-turbo (served) | 0.848 | 0.043 | +0.28 | -0.02 |
| whisperx large-v3 | 0.838 | 0.027 | -0.08 | -0.10 |
| large-v3 + MMS alignment | 0.834 | 0.007 (+ASR) | -0.06 | -0.10 |
| faster-whisper large-v3 | 0.824 | 0.111 | +0.36 | +0.12 |
| parakeet-tdt-0.6b-v3 | 0.818 | 0.004 | +0.12 | -0.36 |
| qwen3-asr-1.7b + forced aligner | 0.816 | 0.084 | -0.02 | -0.18 |
| large-v3 + qwen forced aligner | 0.814 | 0.002 (+ASR) | -0.02 | -0.18 |
| parakeet-tdt-0.6b-v2 + MMS | 0.811 | 0.007 | -0.06 | -0.10 |
| HF whisper-large-v3 | 0.796 | 0.123 | +0.26 | -0.16 |
| parakeet-tdt-0.6b-v2 | 0.794 | 0.004 | +0.06 | -0.36 |
| HF whisper + distil speculative | 0.751 | 0.079 | +0.16 | -0.40 |
| granite-speech 4.1 2b-plus | 0.421 | 0.466 | +1.30 | -1.70 |

Turbo's end fingerprint holds on the cluster run too: 49 percent of annotated ends within 20 ms of a turbo word end, against at most 15 percent for any other model. Parakeet v3 and the two forced aligners are far faster but their ceilings sit 0.03 below turbo. Granite's timestamps are unusable. faster-whisper large-v3 and distil failed on the A10 node (libcublas.so.12 missing from venv-asr there; the large-v3 transcripts already existed) and canary-qwen failed on a torchao import; neither matters for the decision. Decision: the ASR stays large-v3-turbo; the remaining lever is the answering model (job 29429655, pending on the H100 queue).
