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
