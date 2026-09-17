# Committee report 1: open-weights ASR with timestamps (verified 2026-09-17)

WER = Open ASR Leaderboard English mean (8 sets, Whisper normalizer, H200) unless noted. RTFx = leaderboard batched throughput, not single-clip latency. "Native TS" = timestamps produced by the model itself.

| Model | Params / License | Timestamps (mechanism) | WER avg / AMI / Earnings22 | RTFx | MP3 in |
|---|---|---|---|---|---|
| [Parakeet-TDT-0.6B-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (May 2025, EN-only) | 0.6B, CC-BY-4.0 | Native word/segment/char from TDT duration head, 80 ms frame | 6.05 / 11.16 / 11.15 | 3386 | wav/flac 16k mono, decode with ffmpeg |
| [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (Aug 2025, 25 langs) | 0.6B, CC-BY-4.0 | same | 6.34 / 11.31 / 11.42 | 3333 | same |
| [Parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) (Apr 2026) | 0.6B, NVIDIA Open Model Lic. | RNN-T, offline+streaming (TS support not stated on card) | - / 10.14 / 11.16 | n/a | wav |
| [Canary-1B-Flash](https://huggingface.co/nvidia/canary-1b-flash) (Mar 2025) | 0.88B, CC-BY-4.0 | Native timestamp tokens (experimental); F1 95.5/93.5 @200 ms collar on LS | 6.35 / 13.11 / 12.79 | 1046 (A100) | wav/flac; 40 s chunks |
| [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) | 2.5B, CC-BY-4.0 | **None** (confirmed in HF discussion) | 5.63 | 418 | wav |
| [Granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) (Apr 2026) | 2B, Apache-2.0 | Native `[T:N]` centisecond tokens; AAS 38.8 ms EN (vs WhisperX 89.2, Canary-v2 105) | 5.71 / 8.63 / 8.68 (base non-plus: 5.33 but no TS) | n/a (LLM decoder, slow) | array via transformers/vLLM >=0.23 |
| [Granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) (Aug 25 2026) | 0.47B, Apache-2.0 | **No timestamps, no punctuation/caps** exposed (CTC, so frame alignment is derivable) | 5.00 (IBM-reported, unverified) | 12,600 (IBM) | array |
| [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) + [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) (Jan 2026) | 1.7B+0.6B, Apache-2.0 | External NAR LLM aligner; AAS 27.8 ms human-labelled / 42.9 ms MFA-raw; <=5 min | 5.76 / GigaSpeech 8.45 | 0.6B: RTF 0.064 @128 conc. (vLLM) | librosa (mp3 via audioread/ffmpeg) |
| [Cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 2B, Apache-2.0 | **None** | 5.42 | n/a | - |
| [MOSS-Transcribe-preview-2B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B) | 2.4B, Apache-2.0 | **None** | 4.87 / 8.37 / 7.84 | 173 | wav 16k |
| [ARK-ASR-3B](https://huggingface.co/Edge0/ARK-ASR-3B) (May 2026) | 3B, Apache-2.0 | **None**; 30 s max | 5.04 (card) / 8.79 / 8.23 | 491 | wav 16k |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | 1.55B, MIT | Cross-attention DTW (100-400 ms variance) or external aligner | 7.44 / 15.95 / 11.29 | 145 | faster-whisper: PyAV bundled |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0.81B, MIT | DTW, [broken on clips <8 s, repetitions](https://github.com/huggingface/transformers/issues/37248) | 7.83 / 16.13 / 11.63 | 200 | same |
| [distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) | 0.76B, MIT | DTW; 1.46x turbo speed | ~7.1 / 14.63 / 11.29 (card) | - | same |
| [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) | Whisper-large base, **weights non-commercial** | Supervised cross-attention; 29.6 ms boundary error on TIMIT vs WhisperX 64.8 | no WER published | ~Whisper | same |
| [WhisperX](https://github.com/m-bain/whisperx) | wrapper, BSD | wav2vec2 CTC forced alignment | Whisper WER | fast | PyAV |
| [Kyutai stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en) | 2.6B, CC-BY-4.0 | Word-level by stream offset (2.5 s delay), 12.5 Hz frames | LS 6.4 / AMI 12.17 | 88 | 24 kHz mono |
| [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (Feb 2026) | 4B, Apache-2.0 | **Not in open model** (word TS only in API-only Mini Transcribe V2) | FLEURS 4.9 / AMI 15.05 @480 ms | streaming only, vLLM | - |
| [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 5.6B, MIT | None; 40 s max | 6.02 / 11.09 | slow | soundfile |
| Moonshine EN Medium streaming | 0.25B, MIT | not verified | 6.65 (leaderboard) | fast | - |
| Seamless-M4T-v2 | 2.3B, **CC-BY-NC-4.0** | none | FLEURS 6.15 | slow | - |
| [MedASR](https://huggingface.co/google/medasr) (Google, Dec 2025) | 0.1B Conformer, Apache-2.0/HAI-DEF | not documented | 4.6 on radiology dictation vs Whisper 25.3 (dictation, not dialogue) | fast | - |

Note the leaderboard dataset row for ARK-ASR-3B says 4.76 while its card says 5.04; MOSS says 4.87; these Jan-May 2026 LLM-decoder models beat everything above on text but ship no timestamps.

## Ranked recommendation

**1. Parakeet-TDT-0.6B-v2 (primary).** Only model that combines real conversational competence (AMI 11.2), native word timestamps in the same forward pass, permissive CC-BY-4.0, and sub-second latency on a 3-min clip (RTFx >3000 batched; single clip is well under 1 s on any 24 GB GPU). Native TDT timestamps are quantized to 80 ms encoder frames, which is negligible for span IoU at second resolution. Choose v2 over v3 for English-only (v2 is 0.3 WER better and English-specialised). Almost never hallucinates on silence, unlike Whisper.

**2. Qwen3-ForcedAligner-0.6B as a timestamp layer on top of any best-WER text model.** The aligner is Apache-2.0, <=5 min audio, reported 27.8 ms AAS (best published among open aligners), and RTF ~0.001. This decouples the two objectives: use Qwen3-ASR-1.7B (5.76 WER, same toolkit, one call with `forced_aligner=`), or MOSS-Transcribe / Granite-4.1-2b (base) / Cohere-transcribe for the text and align afterwards. With vLLM, 1.7B ASR + aligner should take a few seconds per clip (estimate, not benchmarked). Also lets you align an ensemble/ROVER transcript.

**3. Granite-speech-4.1-2b-plus.** Best WER among models with native word timestamps (AMI 8.63, AAS 38.8 ms), Apache-2.0, vLLM-served. Caveats: timestamp mode is trained only up to 3.5 min (exactly our max; behaviour beyond is unverified), the base 4.1-2b without timestamps is more accurate (5.33) than the plus (5.71), and the LLM decoder emitting a `[T:N]` token per word roughly doubles generation length, so expect several seconds per clip (estimate).

Whisper family ranks below all three for this task: worst conversational WER (AMI ~16), DTW timestamps with 100-400 ms variance and hallucination on silence, turbo has documented broken word timestamps, and CrisperWhisper (the only fix with sub-50 ms accuracy) is non-commercial. Use faster-whisper large-v3 only as a diversity member in an ensemble.

## Concrete pitfalls

- **WhisperX drops timestamps on digits**: wav2vec2 alignment covers [a-z] only, so "500", "mg", "%", are left NaN and interpolated ([issue #869](https://github.com/m-bain/whisperX/issues/869)). For a task where evidence spans are literally about doses, this is disqualifying.
- **Digit vs word output is model-dependent and inconsistent**: Parakeet's ITN "sometimes outputs digits and sometimes words" ([v3 discussion #34](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/34)); LLM-decoder models output digits. Normalize both the transcript and the question text (num2words plus unit expansion: mg/milligrams, mcg/micrograms, bid/twice a day) before matching. Leaderboard WER hides this because the Whisper normalizer maps both forms.
- **Parakeet trailing-silence bug**: a short segment plus silence tail can decode to empty ([NeMo/Speech #15757](https://github.com/NVIDIA-NeMo/Speech/issues/15757)); do not VAD-chunk aggressively, and pass the full 1-3.5 min clip in one pass (v2 handles 24 min).
- **Parakeet segment timestamps** are punctuation-based sentence segments; build evidence spans from word timestamps yourself rather than trusting segment boundaries.
- **MP3**: NeMo cards specify 16 kHz mono wav/flac; decode once with ffmpeg (`-ar 16000 -ac 1`) and feed every model the same array. faster-whisper reads mp3 via bundled PyAV; whisper.cpp needs 16 kHz wav unless built with ffmpeg. Two speakers on one channel is fine for all of these; overlapping speech is where AMI-style WER degradation comes from.
- **Medical vocabulary**: no open dialogue-specific medical evaluation exists; MedASR figures are dictation. Granite 4.1 and Qwen3-ASR support keyword/context biasing prompts, which you can seed with a drug-name list. Parakeet has none; consider a lightweight post-correction lexicon.
- **Licenses to avoid**: CrisperWhisper weights, Seamless (NC), Granite 5.0 "-nc" variant.

## Uncertainty flags

Single-clip GPU latencies above are estimates from RTFx and architecture, not measured. Granite 5.0 TurboCTC WER (5.00) is IBM-only, released 3 weeks ago, no timestamps exposed. Moonshine timestamp support and Parakeet-unified timestamp support were not verifiable from the cards. Parakeet TDT word-boundary error in ms is not published by NVIDIA (only the 80 ms frame tolerance); Granite's AAS comparison table cites Canary-v2 at 105 ms, not Parakeet.

Sources: [Open ASR Leaderboard dataset](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard), [leaderboard blog](https://huggingface.co/blog/open-asr-leaderboard), [Qwen3-ASR report](https://arxiv.org/html/2601.21337v1), [Canary timestamp paper](https://arxiv.org/abs/2505.15646), [Whisper internal aligner](https://arxiv.org/abs/2509.09987), [Voxtral Transcribe 2](https://mistral.ai/news/voxtral-transcribe-2/), [faster-whisper PyAV](https://github.com/SYSTRAN/faster-whisper), [whisper.cpp formats](https://github.com/ggml-org/whisper.cpp/discussions/1399), [Granite 5.0 TurboCTC](https://www.orcarouter.ai/blog/granite-speech-5-0-470m-turboctc-vs-whisper-large-v3-turbo), [MedASR](https://medasr.org/).
