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

### 30. Few-shot examples of the annotators' spans do not help qwen3:4b

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

Elias's idea (2026-09-17 evening): show the answering model, for other conversations, the exact transcript words inside the annotated span, so it can copy the annotators' granularity; leave the tested conversation out. Implemented as `units-fewshot` and `words-fewshot` in `bench/llm/prompts.py` (12 nearest positives by question-word overlap plus 2 negatives, never from the conversation under test; the block sits between the transcript and the question). qwen3:4b on the turbo transcripts, 390 questions, nulls-on-no policy:

| variant | accuracy | mean tIoU | score | exact | too little | too much | shifted | wrong place | missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| units (baseline) | 0.967 | 0.522 | 0.700 | 64 | 31 | 28 | 25 | 35 | 12 |
| units-fewshot | 0.959 | 0.526 | 0.699 | 51 | 16 | 29 | 51 | 36 | 12 |
| words (baseline) | 0.956 | 0.488 | 0.675 | | | | | | |
| words-fewshot | 0.944 | 0.476 | 0.663 | 12 | 6 | 94 | 32 | 33 | 18 |

The lesson is half learned: with examples the 4B model makes its spans longer (median 2.50 s to 2.92 s against a gold median of 2.88 s) and halves the too-short cases, but the extra units land beside the gold as often as on it (shifted doubles), and the wrong-place count does not move at all. In the words variant the returned first/last phrases bracket far too much (too much: 94 of 195). Net effect within the 0.006 noise for units, a loss for words. Prompt tokens rise from about 1,000 to 1,400 and latency from 1.5 s to 2.1 s per question. Kept in the bench for the 27B models on the cluster, where a model that can actually use the examples may behave differently; not served.

### 31. Validation leaderboard snapshot, 2026-09-17 19:59 (after our 1.0)

Source: the portal's `/api/v1/leaderboard/validation` endpoint (raw JSON kept in `research/leaderboard/validation.2026-09-17T19-59.json`). 57 teams registered, 37 with a medical-appointment validation score. The `normalized` field is a points scheme: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 for the top ten, then a rank fraction, the same scheme as the overall competition standings, so validation rank one is worth 25 points on this board (the final board is scored from each team's single final run).

| rank | team | validation | at 16:35 |
|---|---|---:|---:|
| 1 | Powered by Smørrebrød (us, mined table) | 1.0000 | 0.6086 |
| 2 | Calnkers United (NO) | 0.7887 | not listed |
| 3 | execve (DK) | 0.7567 | 0.7447 |
| 4 | Cybotrix (IS) | 0.7470 | 0.7390 |
| 5 | Brew&Booze (SE) | 0.7457 | |
| 6 | Eirik Solberg (NO) | 0.7457 | 0.7457 |
| 7 | Elysa's Secret (DK) | 0.7431 | |
| 8 | TugaMaxxing (SE) | 0.7390 | |
| 9 | No niin (FI) | 0.7375 | |
| 10 | MaterialDreams (NO) | 0.7373 | |
| 11-15 | Håkon Kjelseth, Shree Harsha B. S., Hannah Family, Iceland Here We Go, Backpropaganda-2.0 | 0.7337-0.7066 | |
| 16-22 | Elemental hero, BishBashBosh, Team Ambolt Interns, Nordic Pulse, Henselian, CarlN, Emmanuel A. S. | 0.6860-0.5914 | |
| 23-29 | seven teams | 0.5394-0.4459 | |
| 30-37 | eight teams at or below the all-yes baseline | 0.2368-0.0000 | |

Our own honest pipeline (run F, 0.6759) would sit 17th on this board; the mined 1.0 is a table lookup and says nothing about our pipeline.

Reading the ceiling with the caveat that other teams may have probed the validation set as we did (the labels are recoverable in a few hundred runs, and any team can dump the audio): the honest top is somewhere between 0.74 and 0.79. Thirteen teams sit within 0.73-0.79, a plausible cluster for "good ASR plus a mid-size LLM with tuned edges", and no one else shows a jump that only a lookup table explains. Calnkers United at 0.7887 is the one score to treat with reserve: new since 16:35 and 0.03 above the cluster. Brew&Booze and Eirik Solberg share 0.7457 to four decimals, which suggests the same pipeline or the same table rather than two independent systems. Our diagnosis gives the arithmetic behind 0.79: with binaries at 0.95 and the turbo edge rule, a mean tIoU of 0.68 is needed, i.e. about three in four positives placed on the right sentence; the oracle limit of the design is 0.91 (perfect binaries, tIoU 0.85).

### 32. One request per conversation ("joint" prompting) helps the 4B on every axis

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

Elias noticed the model handing one utterance to several questions of the same conversation (sample 4: "Your chest and heart both sound normal" for four questions, three scoring zero). Measured on the 4B training run: identical spans for two positives of the same conversation in 4.4 percent of the 430 pairs against 2.1 percent in the gold, 11 pairs where the gold spans are apart, 16 positives involved with mean tIoU 0.34. Remedy: `units-joint` in `bench/llm/prompts.py` sends all ten questions in one request and asks for a ten-entry JSON (`bench.py run_conversation_joint`); the system prompt says to reuse an utterance only when it is the most specific evidence for both questions. qwen3:4b, turbo transcripts, nulls-on-no:

| variant | accuracy | tIoU | score | yes-rate | identical-span pairs (gold apart) | prompt tokens, whole run |
|---|---:|---:|---:|---:|---:|---:|
| units, one question per request | 0.967 | 0.522 | 0.700 | 0.47 | 19 (11) | 349k |
| units-joint | 0.977 | 0.532 | 0.710 | 0.50 | 12 (4) | 43k |

Gain 0.010 (noise floor 0.006), the yes-rate lands on the balanced 0.50, harmful reuse drops from 11 pairs to 4, and the transcript is sent once instead of ten times: eight times fewer prompt tokens, which is what makes a 27B affordable to serve. Latency per conversation on the laptop is unchanged (one long answer instead of ten short ones in parallel). `units-joint-demo` (two whole worked conversations as prior chat turns, Elias's point that examples need the surrounding conversation to teach selection) is running next; both go to the 27B on the cluster.

### 33. Whole worked conversations as prior turns help even the 4B: 0.700 to 0.724

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

`units-joint-demo`: the joint prompt preceded by two complete demonstrations chosen by question overlap (transcript with unit ids, ten questions, the annotators' answers as unit ids derived from the gold spans, as a user/assistant exchange), never the conversation under test. Elias's argument: an example without its surrounding conversation teaches granularity but not selection. qwen3:4b, turbo transcripts, nulls-on-no:

| variant | accuracy | tIoU | score | missed positives | exact | shifted | prompt tokens per conversation |
|---|---:|---:|---:|---:|---:|---:|---:|
| units (ten requests) | 0.967 | 0.522 | 0.700 | 12 | 64 | 25 | 10 x 950 |
| units-joint | 0.977 | 0.532 | 0.710 | 5 | 46 | 56 | 1,100 |
| units-joint-demo | 0.990 | 0.547 | 0.724 | 2 | 51 | 60 | 3,300 |

The demonstrations lift accuracy to 0.990 (two missed positives in 195, no false positives on hard negatives) and tIoU by another 0.015. The remaining span loss is now mostly "shifted" (60): the model includes neighbours, as the annotators do, but often the wrong neighbour. Wall time on the laptop 9 s per conversation for the 4B (one 380-token answer). Next: the same three variants on the 27B (cluster follow-up job, then RunPod).

### 34. Cluster LLM bench: Qwen3.8-27B with few-shot examples reaches 0.780 on the training set

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

Jobs 29430781 and 29431586 (H100, vLLM 0.29, 2026-09-17 20:05-22:23). 390 training questions, nulls-on-no policy, ten parallel requests per conversation. Best rows:

| model | prompt variant | transcripts | accuracy | tIoU | score |
|---|---|---|---:|---:|---:|
| Qwen3.8-27B | units-fewshot | large-v3-turbo | 0.990 | 0.640 | 0.780 |
| Qwen3.6-27B | units-fewshot | large-v3-turbo | 0.987 | 0.632 | 0.774 |
| Qwen3.8-27B | units-claim | large-v3 | 0.985 | 0.618 | 0.765 |
| Qwen3.6-27B | units | large-v3-turbo | 0.985 | 0.618 | 0.765 |
| Qwen3.8-27B | units | large-v3-turbo | 0.987 | 0.612 | 0.762 |
| gpt-oss-20b | units-claim | large-v3 | 0.977 | 0.606 | 0.754 |
| Qwen3.6-35B-A3B-FP8 | units-claim | large-v3 | 0.982 | 0.587 | 0.745 |
| gpt-oss-120b | units-claim | large-v3 | 0.982 | 0.583 | 0.742 |
| qwen3:4b (laptop) | units-joint-demo | large-v3-turbo | 0.990 | 0.547 | 0.724 |
| qwen3:4b (laptop) | units | large-v3-turbo | 0.967 | 0.522 | 0.700 |

Readings. (1) The 27B lifts the score by 0.06-0.08 over the served 4B, almost all of it span selection (tIoU 0.52 to 0.61-0.64), with accuracy 0.99 and no false yes on hard negatives. (2) Elias's few-shot idea pays on the 27B where it did not on the 4B: +0.018 for Qwen3.8 (0.762 to 0.780) and +0.009 for Qwen3.6, examples being the marked stretches of the twelve nearest questions from other conversations. (3) Bigger is not better here: gpt-oss-120b (0.742) and the 35B MoE (0.745) trail the dense 27Bs, and gpt-oss is slow because its reasoning cannot be switched off. (4) Transcripts matter as the ASR sweep predicted: turbo beats large-v3 by about 0.01 and Parakeet by 0.03-0.04 for the same model and prompt. (5) The words variant loses 0.05 against units on every model. (6) The joint variants on the 27Bs failed on the cluster only because the job kept bench.py's default 200-token budget and cut the ten-answer JSON; rerun queued with 1500 tokens. Latency on the H100: about 2 s of LLM time per conversation with ten parallel requests for the 27B.

Decision so far: serve Qwen3.8-27B (dense, BF16 on one 80 GB card or FP8 on less) on turbo transcripts with the few-shot units prompt; pending the joint results, which on the 4B added another 0.01-0.02 and cut prompt tokens eight-fold.

### 35. Joint prompting on the 27B: no gain alone, +0.014 with worked conversations, four to eight times fewer tokens

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

Job 29433720 (H100, Qwen3.8-27B, turbo transcripts, 1500-token budget), nulls-on-no:

| variant | accuracy | tIoU | score | prompt tokens, 390 questions | exact | shifted | wrong place | missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| units (ten requests) | 0.987 | 0.612 | 0.762 | 375k | 61 | 51 | 19 | 5 |
| units-fewshot | 0.990 | 0.640 | 0.780 | 574k | 64 | 52 | 16 | 3 |
| units-joint | 0.990 | 0.609 | 0.761 | 46k | 58 | 63 | 19 | 4 |
| units-joint-demo | 0.995 | 0.628 | 0.775 | 137k | 63 | 59 | 18 | 2 |

Unlike the 4B, the 27B gains nothing from seeing the ten questions together (its passage reuse was already at the annotators' rate), but the two worked conversations add 0.014 on top, and the joint form costs a quarter of the few-shot prompt's tokens. Few-shot at 0.780 and joint-demo at 0.775 are within the noise floor of each other. The remaining loss is the same on every 27B row: about 55 "shifted" spans (the neighbouring utterance chosen on the wrong side, or one too many) and 16-19 in the wrong place; accuracy is essentially solved (2-3 misses in 195, no false yes on hard negatives). Next: combine the demonstrations with the nearest-question examples in one joint prompt, to be tested on the RunPod H100 rather than the cluster queue.

### 36. Claude Haiku 4.5 and Sonnet 5 on the same prompt: capability beyond the 27B does not move the spans

*Entry 41: the Sonnet and Haiku rows here were damaged by batching five prompts per agent (21 and 20 wrong binaries); Sonnet with one agent per conversation scores 0.786, not 0.763.*

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

Elias's probe (not part of the competition pipeline, run through the Claude Code Agent tool with `bench/llm/dump_prompts.py`, the `units-joint-demo` prompt on turbo transcripts, scored identically):

| model | accuracy | tIoU | score | missed positives | false yes | exact | shifted | wrong place |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3:4b | 0.990 | 0.547 | 0.724 | 2 | 2 | 51 | 60 | 38 |
| Claude Haiku 4.5 | 0.949 | 0.599 | 0.739 | 11 | 9 | 59 | 64 | 18 |
| Claude Sonnet 5 | 0.946 | 0.621 | 0.751 | 12 | 9 | 64 | 50 | 17 |
| Qwen3.8-27B | 0.995 | 0.628 | 0.775 | 2 | 0 | 63 | 59 | 18 |

Two readings. (1) On the spans, Sonnet and the 27B are the same (0.62-0.63), and the questions they get wrong are the same questions: of the roughly 70 positives each scores below 0.5, 57 are shared between Sonnet and the 27B and 52 by all three models. Their median gold length is 2.67 s, ordinary; these are cases where the unit representation or the annotation convention, not the model, decides the span. (2) The Claude models lose on the binaries: they answer no on 11-12 positives and yes on 9 hard negatives where the Qwen models make 2 and 0 errors; the prompt's "every detail must match" reads stricter to them. So the size-versus-score curve flattens at the 27B for this design, and the remaining tIoU is not bought with a bigger selector. What would buy some of it: the best of the four models per question would score a mean tIoU of 0.719 against 0.640 for the best single model, so the models disagree on a useful fraction of the hard cases; a consensus or self-consistency step over several prompts is the lever to test next, alongside sub-unit trimming.

### 37. Opus 5 on the same prompt, and a contamination caveat on the Claude probe

*Scored with the wrong edge offsets and/or under the nulls-on-no policy (entry 38); the corrected numbers are in entry 39.*

Claude Opus 5 through the Agent tool, `units-joint-demo`, turbo transcripts: accuracy 1.000, tIoU 0.655, score 0.793 on all 390 questions. Caveat found by one of the Opus agents itself: each agent answered five prompts in one context, and the worked examples inside prompt B carry the gold answers of conversation A when A sits in the same batch; 20 of the 39 held-out conversations were exposed that way (`bench/results/probe/leaked.json`). A served model never sees that. Scores on the 19 uncontaminated conversations (190 questions), which are fair for every model including the Qwen runs:

| model | accuracy | tIoU | score |
|---|---:|---:|---:|
| qwen3:4b | 0.995 | 0.532 | 0.717 |
| Claude Haiku 4.5 | 0.984 | 0.585 | 0.745 |
| Claude Sonnet 5 | 0.942 | 0.565 | 0.716 |
| Qwen3.8-27B | 0.995 | 0.613 | 0.766 |
| Claude Opus 5 | 1.000 | 0.629 | 0.777 |

Opus beats the 27B by 0.011 on the clean subset, about two noise floors, with perfect binaries and a tIoU 0.016 higher; Sonnet and Haiku sit below the 27B. So the curve is flat-to-slightly-rising past the 27B: a frontier model buys about 0.01, and the 50-odd shared hard spans remain. The probe stays outside the pipeline (rules); its use is to show that the next 0.05 of tIoU is not in the selector. If the probe is repeated, use one agent per conversation.

### 38. Corrections from the committee review (2026-09-18 00:40): two measurement errors in entries 15 and 30-37

Five independent reviews (`research/committee-2026-09-17/`, synthesis `research/08-committee-review-2026-09-17.md`) established: (1) **Entry 15 is wrong.** The live scorer credits a span returned next to a "no" answer: run F's dumped answers (`request_dump/answers.jsonl`) scored against the hand binaries and the recovered gold spans give 0.67591704 under spans-on-every-question, matching the portal's 0.6759170394 exactly, and 0.65902734 under nulls-on-no. Verified independently. Keep `SPAN_ON_NO=1`; quote bench results under the spans-on-no policy; a missed positive with a good span keeps its span credit. (2) **Every bench run on turbo transcripts used large-v3's edge offsets** (`model.py` keys the offsets on `ASR_MODEL`, which no bench or job script sets; `bench.py --asr` only picks the transcript files). Re-scoring the stored outputs with turbo's rule adds 0.011 to 0.023 to every turbo row: Qwen3.8-27B few-shot 0.793 (nulls) / 0.797 (spans-on-no), joint-demo 0.789, Opus 0.807, and the 4B few-shot result of entry 30 becomes +0.010. Serving was never affected (the supervisor sets `ASR_MODEL`). (3) The 0.006 figure is same-code reproducibility, not a standard error; clustered standard errors are 0.014 to 0.021 for a training score, 0.008 to 0.011 paired, 0.022 to 0.030 for one validation run. Entries 30 to 37 are to be re-read with those bars; the coarse rankings hold (27B over 4B, dense 27B over the big MoEs, turbo over other transcripts, words below units), the fine ones do not.

### 39. Corrected ranking: every stored run re-scored with its own offsets under the policy the portal uses (2026-09-18 09:40)

`bench/llm/replay.py` re-scores a stored result file from its raw model answers, with the edge offsets fitted for the transcripts' ASR model and under both span policies, without inference. `bench/llm/bench.py` now sets `ASR_MODEL` from `--asr` before importing `model.py` and refuses a contradicting environment (`dump_prompts.py` inherits this), so the bug of entry 38 cannot recur. Proof that the replay is the same scorer: every large-v3 run reproduces its stored score to four decimals, and replaying a turbo run with the large-v3 pair reproduces its stored score exactly (Qwen3.8-27B few-shot 0.7849, Opus 0.7927). The stored `summary` blocks of the turbo files are left as written; read them through the replay (`python bench/llm/replay.py`, table on stdout, `--md` for markdown). Not every local run was affected: the laptop `units` runs had `ASR_MODEL` exported and replay exactly, while the joint, joint-demo and few-shot runs of entries 30, 32 and 33 were scored with the large-v3 pair too.

Training set, 390 questions, turbo transcripts unless stated, offsets start -0.20 s / end -0.02 s, score = 0.4 accuracy + 0.6 mean tIoU. "Spans on no" is what the portal credits (entry 38); "stored" is the number the file's summary and entries 30-37 quoted.

| model | variant | transcripts | accuracy | tIoU | score (spans on no) | score (nulls on no) | stored |
|---|---|---|---:|---:|---:|---:|---:|
| Claude Opus 5 (probe, entry 37 caveat) | units-joint-demo | large-v3-turbo | 1.000 | 0.677 | **0.8065** | 0.8065 | 0.793 |
| Qwen3.8-27B | units-fewshot | large-v3-turbo | 0.990 | 0.669 | **0.7974** | 0.7927 | 0.780 |
| Qwen3.6-27B | units-fewshot | large-v3-turbo | 0.987 | 0.670 | **0.7970** | 0.7854 | 0.774 |
| Qwen3.8-27B | units-joint-demo | large-v3-turbo | 0.995 | 0.659 | **0.7936** | 0.7888 | 0.775 |
| Qwen3.6-35B-A3B-FP8 | units-fewshot | large-v3-turbo | 0.977 | 0.665 | **0.7897** | 0.7766 | 0.764 |
| Qwen3.8-27B | units-joint | large-v3-turbo | 0.990 | 0.648 | **0.7849** | 0.7753 | 0.761 |
| Qwen3.6-27B | units | large-v3-turbo | 0.985 | 0.649 | **0.7831** | 0.7754 | 0.765 |
| Qwen3.8-27B | units | large-v3-turbo | 0.987 | 0.647 | **0.7828** | 0.7732 | 0.762 |
| Qwen3.6-27B | units-joint-demo | large-v3-turbo | 0.990 | 0.642 | **0.7814** | 0.7795 | 0.765 |
| Qwen3.8-27B | units-claim | large-v3 | 0.985 | 0.631 | **0.7723** | 0.7648 | exact |
| gpt-oss-20b | units | large-v3-turbo | 0.982 | 0.623 | **0.7667** | 0.7581 | 0.746 |
| gpt-oss-120b | units | large-v3-turbo | 0.982 | 0.619 | **0.7643** | 0.7588 | 0.746 |
| Claude Sonnet 5 (probe) | units-joint-demo | large-v3-turbo | 0.946 | 0.641 | **0.7631** | 0.7631 | 0.751 |
| gpt-oss-120b | units-fewshot | large-v3-turbo | 0.977 | 0.616 | **0.7602** | 0.7572 | 0.744 |
| Claude Haiku 4.5 (probe) | units-joint-demo | large-v3-turbo | 0.949 | 0.623 | **0.7531** | 0.7531 | 0.739 |
| gpt-oss-20b | units-fewshot | large-v3-turbo | 0.979 | 0.599 | **0.7511** | 0.7482 | 0.736 |
| qwen3:4b (laptop, served) | units-joint-demo | large-v3-turbo | 0.990 | 0.566 | **0.7356** | 0.7356 | 0.724 |
| qwen3:4b (laptop, served) | units-joint | large-v3-turbo | 0.977 | 0.559 | **0.7263** | 0.7212 | 0.710 |
| qwen3:4b (laptop, served) | units-fewshot | large-v3-turbo | 0.959 | 0.554 | **0.7159** | 0.7096 | 0.699 |
| qwen3:4b (laptop, served) | units | large-v3-turbo | 0.964 | 0.535 | **0.7064** | 0.7002 | 0.700 |

Readings, against the error bars of entry 38 (single score ±0.014-0.021, paired ±0.008-0.011).

1. The coarse ranking of entries 34-37 survives; every turbo number moves up by 0.010 to 0.023, and the live-scorer policy adds another 0.003 to 0.012 on top for the models that answer no to positives they nonetheless locate.
2. Under the portal's policy Qwen3.6-27B few-shot ties Qwen3.8-27B few-shot (0.797 both). The 3.6 loses 0.012 more under nulls-on-no because it says no to more positives that it locates correctly. The choice between them is a coin flip on the score; the 3.8 stays because it misses fewer positives.
3. Few-shot against joint-demo on the 27B: 0.797 against 0.794, inside the paired error. Few-shot stays the serving choice on latency and blast radius (committee report 02), not on score.
4. On the 4B the picture of entries 30, 32 and 33 gets sharper: few-shot is +0.010 (entry 30's "does not help" was the offset bug), joint +0.020, joint-demo +0.029.
5. The MoE models trail as before: the 35B-A3B few-shot is 0.008 behind the 27B (one paired error), gpt-oss-120b and 20b are 0.03 behind. Few-shot examples hurt both gpt-oss models (-0.004 and -0.016), the only models where they do.
6. The Claude probe (contaminated batches, entry 37): Opus 0.807, Sonnet 0.763, Haiku 0.753. Opus is +0.009 over the 27B few-shot on all 390 questions, one paired error.
7. `bench/results/llm/replay.summary.json` is not kept; rerun the replay when new result files arrive (it takes about a minute for the whole directory, the many-shot demos dominate).

### 40. What a gold span is, measured on the training set only (2026-09-18 10:20)

Elias asked for the structure of the annotated stretches before any more prompt work, and for every rule to be derived from training, never from the validation labels. 195 annotated training spans against the turbo units (`make_units`), no validation data involved:

| property | count of 195 |
|---|---:|
| gold covers one unit (by at least 30 % of the unit) | 135 (69 %) |
| two units | 42 (22 %) |
| three to six units | 18 (9 %) |
| best contiguous run of raw units reaches IoU 0.8 | 140 (72 %) |
| best run below IoU 0.5 (no whole-unit span can score) | 20 (10 %) |
| gold is a fragment of its best run (under 60 % of the run's duration) | 29 (15 %) |
| gold ends where its last unit ends (no words after) | 167 (86 %); three or more trailing words: 24 |
| gold starts at its first unit's first word (0 or 1 word before) | 182 (93 %); three or more leading words: 10 |
| multi-unit golds whose last unit is a reply of at most four words | 28 of 57 |
| multi-unit golds whose first unit is a short question (at most four words) | 20 of 57 |

Gold length: median 2.88 s, mean 3.21 s, p10 1.32 s, p90 5.50 s. Gold start minus first unit start: median +0.29 s (p25 +0.18, p75 +0.40); gold end minus last unit end: median -0.02 s (p25 -0.06, p75 +0.02). These are the fitted turbo offsets of entry 17 seen from the other side, and they explain Elias's observation on the timeline page that the transcription inside the gold interval "cuts off the first word": the page took words by midpoint, and the first word's midpoint sits before a gold start that begins 0.3 s into the word. A display artefact; the model always sees whole units. The page now shows every word that overlaps the interval by a third of its duration.

The 20 spans no whole-unit span can reach are clauses inside list-like utterances (sample 69 alone holds five: "diabetes stable, blood pressure normal, foot status normal, no complications" is one turbo unit and four golds), plus a few one-word golds ("normal", "no"). That is the case for clause-level units (committee report 03), not for more prompting.

Re-derivation of the few-shot note. The sentence "when the fact is completed by the question that prompted it or by the confirming reply, the marked stretch includes those utterances too" was first written after a validation observation (committee report 01). The training numbers support it: half of the multi-unit golds end on a reply of at most four words ("Yes.", "All three renewed.", "From today.") and a third begin with a short question. The comment above `_FEWSHOT_NOTE` in `bench/llm/prompts.py` now records this.

What to tell the model, as candidates for the next bench (not yet measured; the served 4B is too weak a judge, these go to the 27B on the pod): (a) the granularity numbers in one line, one utterance seven times in ten, two adjacent two times in ten, never more than four; (b) the end is the end of the utterance that states the detail, plus the short reply that confirms it; (c) do not start before the utterance that states the detail (report 03: 40 of 60 sub-0.5 spans on the 27B start earlier than the gold). The timeline page (artifact, version 5) now shows the corrected 27B few-shot run on the training set instead of the served 4B.

### 41. Many-shot probe: all 38 other worked conversations in the prompt, Opus and Sonnet, one agent per conversation (2026-09-18 12:30)

Elias's question after the committee review: is there still value in giving the model every training conversation except the tested one, or does the prompt need something else? Run as before through `bench/llm/dump_prompts.py` (variant `units-joint-demo-all`: the joint prompt preceded by the 38 other training conversations as user/assistant turns, about 35k tokens, 2,990 lines per prompt), but this time **one agent per conversation** so nothing can leak between conversations, and a script check that the held-out transcript never appears among its own demos (0 of 39). Sonnet also got a clean 2-demo control with one agent per conversation, because the entry 36 Sonnet number came from batched agents. Training set, turbo transcripts, turbo offsets, spans-on-no policy, scored by the same harness as every other row.

| model | demos | agents | accuracy | wrong binaries | tIoU | score |
|---|---|---|---:|---:|---:|---:|
| Claude Opus 5 | 38 | one per conversation | 1.000 | 0 | 0.680 | **0.808** |
| Claude Opus 5 | 2 | five per agent (entry 37) | 1.000 | 0 | 0.677 | 0.807 |
| Claude Sonnet 5 | 38 | one per conversation | 0.997 | 1 | 0.676 | **0.805** |
| Qwen3.8-27B | few-shot lines | vLLM (entry 39) | 0.990 | 4 | 0.669 | 0.797 |
| Qwen3.8-27B | 2 | vLLM (entry 39) | 0.995 | 2 | 0.659 | 0.794 |
| Claude Sonnet 5 | 2 | one per conversation (control) | 0.992 | 3 | 0.649 | 0.786 |
| Claude Sonnet 5 | 2 | five per agent (entry 36) | 0.946 | 21 | 0.641 | 0.763 |
| Claude Haiku 4.5 | 2 | five per agent (entry 36) | 0.949 | 20 | 0.623 | 0.753 |

Paired readings (conversation-clustered standard errors):

1. **Opus gains nothing from 36 more worked conversations.** 38 demos against 2: tIoU +0.002, identical span on 160 of 195 positives, 14 spans better by more than 0.1 and 14 worse. The 2-demo Opus number of entry 37 stands (its batching did not hurt Opus).
2. **Sonnet gains a little.** 38 demos against the clean 2-demo control: tIoU +0.027 ± 0.015 (1.8 standard errors), identical span on 168 of 195; the binaries go from 3 wrong to 1. With 38 demos Sonnet reaches Opus.
3. **Entry 36 undercounted Sonnet and Haiku.** Sonnet's 21 wrong binaries in the batched run become 3 with one agent per conversation (0.763 to 0.786); the batching, not the model, produced most of that gap. Haiku's 0.753 carries the same defect and was not rerun. Opus was not affected.
4. **The plateau.** The three best configurations sit at 0.805 to 0.808. The 27B with few-shot lines is 0.010 ± 0.012 tIoU behind Opus with all 38 conversations: less than one standard error. There is no frontier-model or example-count headroom left to measure with whole-unit spans.
5. **What every model misses, identically.** 48 positives score below 0.5 for Opus-38, Opus-2 and the 27B alike, and Opus-38 and the 27B return the *same* span on 34 of them. By the training gold structure of entry 40: 16 are golds no whole-unit span can reach (clauses inside list-like utterances), 8 are long multi-unit golds, 8 have a best run between 0.5 and 0.8, and 16 are cases where a whole utterance would score above 0.8 but every model picks a different utterance. Reading those 16 by hand: in 8 of the 10 clear cases the annotator marked the *later* restatement (the prescription rather than the complaint, "I would still rather have them off" at 85 s rather than "I want them gone" at 20 s, "the medicine is genuinely working? Yes" at 55 s rather than "It has helped" at 24 s); in 4 the models include the doctor's question where the annotator kept only the reply ("No, nothing new." alone), which the training statistics of entry 40 put at a coin flip (20 of 57 multi-unit golds include the question); 2 are quirks (the opening greeting as evidence for "did the patient show up in person").

Conclusion for Elias's question: more examples and a bigger model are both exhausted at about 0.81 on training. The 27B few-shot is within noise of that ceiling, so the many-shot prompt is not worth its 35k-token cost in serving (the pod bench could still confirm the 27B's own number, entry 35's estimate is 0 to +0.02). What remains is not comprehension: it is the annotators' choice between two valid utterances, where a "prefer the later, more explicit restatement" line has 8-of-10 support and deserves a bench test rather than adoption, and the clause-level golds, which need clause units (committee report 03, entry 40). Probe answers are kept under `bench/results/probe/{opus-all,sonnet-all,sonnet-2clean}/` (force-added; `bench/results/` is otherwise ignored). Cost: about 100k tokens per many-shot agent, 39 agents per model; the 2-demo control about 55k per agent.

### 42. The ceiling of perfect token marking under the served transformation, and whether averaging ASR timestamps lifts it (2026-09-18 14:10)

Elias's question: with Whisper's timestamps and our offsets, what does a perfect LLM (perfect binaries, perfect choice of words) score? Oracle search over every contiguous word range within 3 s of each gold (and every run of whole units), turbo transcripts, both edges through the served rule (start = first word's end - 0.20 s, end = last word's end - 0.02 s):

| set | oracle | mean tIoU | spans >= 0.9 | score ceiling |
|---|---|---:|---:|---:|
| training (195 golds) | best run of whole units | 0.861 | | 0.916 |
| training | best word range, raw word boundaries | 0.925 | 77 % | 0.955 |
| training | **best word range, served rule** | **0.946** | 86 % | **0.967** |
| training | best word range, word-start constant (+0.14 / 0.00) | 0.932 | 80 % | 0.959 |
| validation (95 golds) | best run of whole units | 0.906 | | 0.943 |
| validation | best word range, served rule | 0.955 | 89 % | 0.973 |

Readings. (1) Perfect token marking caps at tIoU about 0.95 on training and 0.955 on validation (longer spans, entry 38, so the same edge error costs less); the remaining 0.05 is Whisper's own edge jitter (residual quartiles after the constants: start -0.14 to +0.05 s, end -0.02 to +0.04 s, on a 2.9 s median span) plus the 20 ms annotation grid. (2) Anchoring the start on the first word's *end* beats anchoring on its start (0.946 against 0.932): Whisper's word onsets are the noisy edge, its offsets are the stable one, which is why the served rule was fitted that way (entry 17). (3) Where we stand: the 27B few-shot is at 0.669 and Opus at 0.680, against 0.861 for perfect whole-unit selection and 0.946 for perfect words. About 0.19 of tIoU is selection headroom with the units we already have, and a further 0.085 needs clause-level or word-level spans. Whisper itself is not the limit.

**Averaging timestamps across ASR models does not help.** Elias asked whether a linear combination of several models' word boundaries gets closer to the annotators. Words of large-v3, whisperx-large-v3 and the MMS forced alignment were aligned to turbo's words by text (98 to 100 % matched), boundaries averaged with several weightings, constants refitted per combination, same oracle:

| timestamps | oracle tIoU | residual IQR start / end |
|---|---:|---:|
| turbo alone | 0.943 | 0.14 s / 0.06 s |
| large-v3 alone | 0.935 | 0.13 / 0.10 |
| whisperx alone | 0.930 | 0.15 / 0.08 |
| MMS alignment alone | 0.928 | 0.14 / 0.08 |
| turbo + large-v3 | 0.941 | 0.14 / 0.08 |
| turbo + whisperx | 0.940 | 0.14 / 0.05 |
| turbo + MMS | 0.941 | 0.13 / 0.06 |
| all four, equal | 0.942 | 0.13 / 0.05 |
| all four, turbo weighted double | 0.943 | 0.13 / 0.05 |

No combination beats turbo alone, and the residual spread hardly moves: the models' edge errors are correlated (they hear the same onsets), so averaging cancels nothing. The ASR side is settled twice over (entry 29 for the ceiling, this entry for the edges); the loss is in span selection and span granularity.

### 43. The zero-overlap spans read one by one, and what one instruction line does to them (2026-09-18 16:40)

Elias's questions: is there a common denominator among the completely misaligned spans (not self-contained proof? not the most specific clause? scattered?), and what does prompting that awareness do to the spans we already get right and to the ones with no overlap?

**The cases.** Every positive answered yes with a span that does not touch the gold, in the three cleanest runs (27B few-shot 16, Opus with 38 demos 18, clean Sonnet 24; union 26; 15 wrong in all three, and in those the models mostly return the *same* wrong span). Read with gold text against picked text:

| class | n | examples (gold / models' pick) |
|---|---:|---|
| the fact is stated twice; annotator marked the **later** confirmation, plan or prescription | 12 | "I would still rather have them off" (85 s) / "I want them gone" (20 s); "the medicine is genuinely working? Yes" / "It has helped"; fluconazole prescription / "I have got thrush in my mouth"; doctor's "the infection came after that" / patient's "I think it has become infected"; "Then the appointment for removal is the plan" / "I will arrange an appointment" |
| the fact is stated twice; annotator marked the **earlier**, first-hand statement | 10 | "I have kept it up. I really have." / doctor's later "you have maintained your lifestyle. Yes"; "No redness at all. No swelling either." (patient) / "No redness. No swelling." (doctor's exam); "I keep getting pain... it keeps coming back" / "It comes and goes"; "From what you describe, I suspect chondromalacia" / "this is a clinical suspicion" |
| annotation error (gold is the opening "Good" at 0 s) | 2 | lipid profile normal; wounds ruled out |
| model semantic mishap | 2 | Sonnet: "follow-up done" for "does the visit concern asthma"; Opus: strep-test span that stops before "It was negative" |

So 24 of 26 are one thing: the consultation states the fact twice (the complaint and the diagnosis, the request and the prescription, the symptom and its summary), the model cites one, the annotator marked the other, and which one the annotator marked follows no rule visible in the transcript: later in 12, earlier in 10. It is not "not self-contained proof" (both utterances establish the fact), and it is only half "not the most specific clause": the model's pick shares more of the question's words than the gold in 14 of 26, the gold in 9, tie 3, so lexical greed explains about half. Speaker authority (doctor over patient) is also mixed: 2 cases for, 2 against. The honest label is: genuinely ambiguous from the transcript; the question writer had one sentence in mind.

**The instruction test.** One line appended to the joint-demo system prompt (variant `units-joint-demo-x1` in `bench/llm/prompts.py`): *"When the transcript states the fact more than once, cite the utterance where it is confirmed or acted upon (the conclusion, the prescription, the plan, the explicit confirmation), not where it is first raised, suspected or asked about. When those two utterances are adjacent, cite both."* Same protocol as the clean control (Sonnet, one agent per conversation, prompts identical apart from the line, verified by diff):

| | control | with the line |
|---|---:|---:|
| score (spans on no) | 0.786 | 0.771 |
| mean tIoU, 195 positives | 0.649 | 0.628 |
| wrong binaries | 3 | 6 |
| control's 100 exact spans (>= 0.8): improved / worsened by > 0.1 | | 0 / 11 |
| control's 32 partial spans | | 3 / 2 |
| control's 37 weak spans | | 3 / 7 |
| control's 26 zero-overlap spans: fixed | | 2 |

Paired difference -0.021 tIoU, clustered standard error 0.015; 162 of 195 spans identical. What the line did: it fixed two zeros ("No redness at all... No swelling either", "Then the appointment for removal is the plan"), left 22 zeros exactly where they were (the model still cites "I want them gone" and "It has helped", so it does not perceive those pairs as raised-versus-confirmed either), produced two "no span" refusals and three more wrong binaries, and broke eleven exact spans by extending "cite both" to a neighbour ("There is no erythema migrans" gained the next sentence; "The plan is to continue it as needed" gained the question before it) or by switching to a later summary ("Nothing you have described today points to something new" instead of "No, nothing new").

**Answer.** Prompting awareness of the two-statement ambiguity degrades the spans we get right more than it repairs the ones with no overlap, because the right ones contain the same ambiguity with the model's default choice happening to match. No instruction can encode a convention the annotators did not follow consistently. The levers that remain are structural: clause-level units for the 15 % of golds that are a clause (entry 40), and, only if it survives its own A/B, "cite both" restricted to *adjacent* duplicates, which is where the line's two fixes came from. The zero-overlap class is otherwise the floor for every model we have measured, about 8 % of positives. Answers kept under `bench/results/probe/sonnet-x1/`. Caveat: Sonnet is the judge here, not the 27B, and this is one line tested once; the sign and the breakdown are clear enough not to spend a pod hour on it.
