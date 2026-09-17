# Committee report 3: word-level timestamps and evidence-span construction (verified 2026-09-17)

## Headline finding (local data, high confidence)

All 390 annotated `evidence_start`/`evidence_end` values in `data/question_train.csv` are exact multiples of **0.02 s** (100%; 20% are multiples of 0.1 s, exactly what a uniform draw on a 20 ms grid gives; three values are float artefacts like `28.520000000000003`). 20 ms is the Whisper token grid and the wav2vec2/MMS CTC frame stride. The annotations were therefore produced by a Whisper-family word/segment-timestamp pipeline (native Whisper, WhisperX or an MMS-style CTC aligner), not by hand on a waveform. Tools on an 80 ms grid (Qwen3-ForcedAligner, Parakeet-TDT native, NFA on FastConformer) cannot hit those edges exactly; MFA's 10 ms grid can, but its "phonetically better" boundaries may differ systematically from the annotator's tool.

Span statistics (n = 195): median 2.88 s, mean 3.21 s, IQR 1.94-4.04 s, 5-95% 1.18-6.78 s, min 0.16 s, max 14.2 s. 30/430 same-conversation span pairs overlap and 9 are identical (several questions share one passage).

## Part A: aligner comparison

| Tool | License | Reported word-boundary precision | Speed | Robust to ASR errors | Digits ("100 mg") |
|---|---|---|---|---|---|
| **WhisperX** (wav2vec2 char CTC, 20 ms) | BSD-2 ([LICENSE](https://github.com/m-bain/whisperX/blob/main/LICENSE)) | TIMIT mean 34 ms / median 23.5 ms, 82% <=50 ms ([Rousso 2024](https://arxiv.org/html/2406.19363)); [Aligner-SUPERB](https://github.com/lifeiteng/Aligner-SUPERB): WBE 46 ms (start 59, end 33); [MFA-2026 paper](https://arxiv.org/html/2606.18466v1) reports 110 ms mean; discrepancy unexplained, likely setup | ~1 s per 3 min on GPU | Forces wrong words onto audio; no wildcard | **No**: dictionary is [a-z]; numbers get no time ([#869](https://github.com/m-bain/whisperX/issues/869), [#98](https://github.com/m-bain/whisperX/issues/98)); fix PR #986 broke last-word alignment ([#1016](https://github.com/m-bain/whisperX/issues/1016)) |
| **ctc-forced-aligner** / **torchaudio MMS_FA** (MMS-300M, 20 ms) | code BSD-2; **model CC-BY-NC-4.0** ([repo](https://github.com/MahmoudAshraf97/ctc-forced-aligner), [torchaudio](https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html)) | Aligner-SUPERB WBE 27 ms (best neural entry), Lhotse-MMS 36 ms; Rousso: TIMIT mean 68 / median 29 ms, Buckeye 41 / 22 ms | ~1 s per 3 min; 5x less memory than torchaudio API | `<star>` wildcard token for untranscribed/wrong stretches | **No** by default: uroman drops digits; pre-normalise with num2words |
| **NeMo Forced Aligner** | Apache-2.0 | WBE 77 ms (Aligner-SUPERB); 78 ms mean, 38% <=50 ms (MFA-2026); AAS 89-130 ms ([Qwen3-ASR report](https://arxiv.org/html/2601.21337v1)); systematic late bias from CTC peaks ([NVIDIA 2025](https://arxiv.org/pdf/2505.15646)) | fast | CTC; no wildcard documented | Undocumented (uncertain); needs CTC/hybrid model: local `parakeet-ctc-0.6b` works, `parakeet-tdt-0.6b-v3` does not ([docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)) |
| **Montreal Forced Aligner 3.x** (10 ms) | MIT | Best on human boundaries: TIMIT mean 20-22 ms, ~90% <=50 ms; Buckeye 22-28 ms (MFA-2026, Rousso) | Kaldi CPU; per-call start-up overhead is a real risk in a 60 s budget (unmeasured) | HMM forced path; no wildcard | **No**: dictionary OOV; must num2words + G2P |
| **Qwen3-ForcedAligner-0.6B** (Jan 2026, 80 ms bins) | Apache-2.0 ([HF](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)) | AAS 42.9 ms avg vs WhisperX 133 / NFA 130 (MFA-labelled); 27.8-32.4 ms on human-labelled; stable on 5-min audio | RTF ~0.001 (vLLM) | LLM slot-filling, NAR; not evaluated on wrong transcripts | **Yes**: tokenizer keeps letters, digits, apostrophes ([code](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py)); but **open bug**: ~2.5% of words get zero-duration spans, 35% of clips affected ([#197](https://github.com/QwenLM/Qwen3-ASR/issues/197)) |
| **Parakeet-TDT-0.6B-v3 native** (80 ms) | CC-BY-4.0 ([HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) | No published boundary accuracy | free with ASR | n/a (it is the ASR) | Emits digits itself |
| **Kyutai STT** | CC-BY-4.0 | Word timestamps native (80 ms Mimi frames), no accuracy published, no forced-align mode ([kyutai.org](https://kyutai.org/stt/)) | streaming | n/a | n/a |
| Others | CrisperWhisper CC-BY-NC (F1@50 ms 0.79 AMI/0.69 TIMIT); [Whisper internal aligner](https://arxiv.org/abs/2509.09987) "better than prior work at 20-100 ms" (numbers not extractable); LLM-ForcedAligner code unreleased; BFA withdrawn Sep 2026 | | | | |

## Recommended alignment stack

1. **ASR**: Parakeet-TDT-0.6B-v3 (already local) for text; keep its digits.
2. **Primary aligner**: ctc-forced-aligner / MMS_FA on a num2words-normalised copy of the transcript ("100 mg" to "one hundred milligrams"), keeping an index map back to the original tokens. 20 ms grid matches the annotations; ~30 ms word error; `<star>` absorbs ASR mistakes and backchannels. (NC licence: fine for a hackathon; if that matters, use WhisperX's BSD English aligner instead.)
3. **Cross-check**: Qwen3-ForcedAligner (Apache) as a second opinion; discard its zero-duration words; use disagreement > 300 ms as a "re-decode this region" flag.
4. **Skip MFA at serve time** (start-up cost, digits), but use it offline to see whether the annotator's edges sit closer to MFA or to wav2vec2 boundaries.

## Recommended span construction

Evidence from tIoU-scored tasks ([TREC MedVidQA 2024](https://arxiv.org/html/2412.11056), [NLPCC 2025](https://arxiv.org/pdf/2505.06814)/[2026](https://arxiv.org/pdf/2607.06618)): winning entries anchor spans on subtitle/ASR timestamps, then apply span expansion, merging of adjacent units and a length prior. End-to-end audio-LLM grounding is far behind (TAG-Bench best mIoU 31, 21% recall at IoU >= 0.7; [paper](https://arxiv.org/html/2609.01542v2)). So:

- **Unit = sentence/clause**, not word phrase and not speaker turn. Median annotated span (2.9 s) is one spoken sentence; 75th percentile (4 s) is one or two. Build units from punctuation in the ASR output, split further at pauses > 0.5 s between word timestamps.
- Select the minimal run of consecutive units that contains the answer's key tokens (drug, number, unit, duration); merge adjacent units only when both are needed. Cap at ~7 s unless the evidence really spans more (95th percentile 6.8 s).
- **Edges = first word start / last word end from the aligner + learned bias.** Fit the median signed offset per edge on the 39 training files. CTC word ends are typically early, so expect a positive end correction (+40-100 ms); verify rather than assume.
- **Pad slightly outward under uncertainty.** For a 2.9 s span, 100 ms over-coverage per edge costs 6.5 IoU points, under-coverage 7 points; at 0.8 s the asymmetry grows (0.78 vs 0.72). Never pad to a turn or the whole file (README: 0.038).
- **First experiment to run** (cheap, decisive): align all 39 files with WhisperX, MMS and native Whisper word timestamps; for each annotated edge, take the nearest word boundary and plot signed offsets. The tool whose offsets cluster at 0 on the 20 ms grid is the annotators' tool; then copy it. Also compute the oracle tIoU of "best sentence unit" to see if sentence granularity caps you (aim > 0.85).

## Pitfalls

- **Whisper native timestamps**: 20 ms token grid but DTW cross-attention drifts by hundreds of ms and across 30 s windows; WhisperX's VAD cut-and-merge exists precisely for this ([WhisperX paper](https://arxiv.org/abs/2303.00747)).
- **MP3 decode offset**: every MP3 decoder adds 528 (+1) samples of delay; LAME adds 576 encoder delay; ffmpeg strips 1105 samples using the LAME/Info tag, torchaudio/minimp3 historically did not ([pytorch/audio #1500](https://github.com/pytorch/audio/issues/1500), [LAME FAQ](https://lame.sourceforge.io/tech-FAQ.txt)). Mismatch vs the annotators' decoder = a fixed ~25 ms shift on every edge. Files are Lavf62 (ffmpeg 8) CBR 44.1 kHz; decode with ffmpeg and check by cross-correlating two decoders.
- **80 ms grids** (Qwen, Parakeet-TDT, NFA) add +/-40 ms quantisation per edge, ~2-3 IoU points on a 3 s span.
- **Digits** silently get interpolated or dropped by every wav2vec2/MFA path; the questions are about doses, so this is the most likely source of large edge errors.
- **Shared passages**: 9 identical span pairs; do not force distinct spans for distinct questions.

Uncertainties: NFA digit handling and MFA start-up latency were not verifiable from docs; the WhisperX numbers disagree across benchmarks by 3x; the 20 ms-grid inference is strong but which Whisper-family tool produced the annotations is unknown until the offset experiment is run.
