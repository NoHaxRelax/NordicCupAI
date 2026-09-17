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
