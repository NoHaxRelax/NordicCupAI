# Committee report 5: audio-native LLMs and prior art for timestamped yes/no QA (verified 2026-09-17)

## Part A: audio-native open-weights models

All facts below were checked against model cards/papers in Sept 2026; "?" marks what could not be verified.

| Model (size) | License | Timestamps / grounding | English WER (LS clean/other unless noted) | Verdict for this task |
|---|---|---|---|---|
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) + [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) (Jan 2026) | Apache 2.0 | Aligner gives word timestamps for any text, up to 5 min; ~43 ms better AAS than WhisperX; aligns digits | 1.63 / 3.38; CV 7.39; RTF ~148x | **Use.** Best ASR+alignment pair for this budget |
| [Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Apache 2.0 | Can emit timestamps when asked, but timestamp-ASR error ~647 ms AAS ([MOSS report](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking)) | 1.22 / 2.48 | No: ~60 GB VRAM at load ([vLLM blog](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization)), coarse timing |
| [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | Apache 2.0 | Zero-shot word alignment 0.1% @20 ms ([frame-level paper](https://arxiv.org/html/2602.10230v1)) | 1.8 / 3.4 | No |
| [Voxtral Mini 3B / Small 24B (2507)](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | Apache 2.0 | No timestamps in chat models; audio QA up to 40 min | Mini 1.86/4.04, Small 1.53/3.14 ([paper](https://arxiv.org/html/2507.13264)) | Maybe as audio-only verifier (9.5 GB / 55 GB) |
| [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (Feb 2026) | Apache 2.0 | Word timestamps advertised for the API "Transcribe V2"; open Realtime model's timestamp output **?** | ~4% FLEURS ([Mistral](https://mistral.ai/news/voxtral-transcribe-2/)) | Backup ASR |
| [MOSS-Audio 4B/8B](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking) (Apr 2026) | Apache 2.0 | Native timestamped ASR (AAS 132 ms on LS); 29.6 mIoU on TAG-Bench | English WER not on card **?** | Most interesting audio-native option; still weak at grounding |
| [FireRedAudio 9B](https://huggingface.co/FireRedTeam/FireRedAudio) (21 Aug 2026) | Apache 2.0 | Best on TAG-Bench, 31.2 mIoU / R@0.7 21.5 | 0.67 / 2.91 (self-reported) | Too new; VRAM/speed **?** |
| [Kyutai stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en) | CC-BY 4.0 | Word timestamps (2.5 s delay) | mean 6.4, AMI 12.2 | Backup ASR only |
| [Step-Audio 2 mini](https://github.com/stepfun-ai/Step-Audio2) (~8B) | Apache 2.0 | None | 1.33 / 2.86 | No |
| [Kimi-Audio-7B-Instruct](https://github.com/MoonshotAI/Kimi-Audio) | MIT | None | 1.28 / 2.42 | No |
| [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) (5.6B) | MIT | "Not supported unless finetuned" | 6.14 OpenASR | No |
| Gemma 3n / [Gemma 4](https://ai.google.dev/gemma/docs/capabilities/audio) E2B/E4B/12B | Gemma / Apache 2.0 | None; **30 s max audio** | n/a | No |
| [Ultravox v0.7](https://huggingface.co/fixie-ai/ultravox-v0_7-glm-4_6) | MIT | None | 2.28 | No: GLM-4.6 355B backbone |
| [Audio Flamingo 3 / AF-Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) | NVIDIA OneWay **Noncommercial** | AF3: 8.7 mIoU TAG-Bench | n/a | No (license, grounding) |
| GLM-4-Voice-9B (2024) | see repo | None | n/a | No |
| Qwen3.5-Omni (Mar 2026), Meta Muse Voice Transcribe (Sep 2026) | API only | - | - | Excluded (not open weights). A blog claim of "Llama 4 405B with audio, open weights Aug 2026" came from a content farm and could not be verified |

**Scepticism, with evidence.** [TAG-Bench](https://arxiv.org/abs/2609.01542) (Sep 2026, 21 systems incl. single-spoken-word queries): best 31.2 mIoU, nine systems under 5 mIoU. [Frame-level tool use](https://arxiv.org/html/2602.10230v1): text-token timestamps hallucinate and collapse out of distribution; mean deviations 0.3-6 s. [Modality arbitration](https://arxiv.org/abs/2602.11488): given audio plus a text transcript, audio-LLMs side with the text even when told to trust the audio, so "re-listen to check the dose" only works if the prompt contains **no transcript**. [Speech-integration study](https://arxiv.org/abs/2512.16378): cascades "remain the most reliable solution overall". [SageLM](https://arxiv.org/pdf/2508.20916): Qwen2.5-Omni trails Whisper+GPT-4o. No paper isolates numeric-detail accuracy of audio LLMs on medical speech; treat that as unknown.

## Part B: prior art

- **MedVidQA 2022** (13 teams, primary metric IoU=0.7, [overview](https://aclanthology.org/2022.bionlp-1.25/)): "video subtitles are dominant features"; every top system did span prediction / sequence labelling with a PLM over timestamped subtitles and mapped the text span back to time. [VPTSL](https://arxiv.org/abs/2203.06667) beat visual span predictors by 28.4% mIoU.
- **MICCAI 2025** ([paper](https://papers.miccai.org/miccai-2025/0978-Paper5153.html)): zero-shot LLM over Whisper-large subtitles, +41% mIoU vs zero-shot multimodal models and above supervised SOTA.
- **NLPCC 2026 Task 1** ([overview](https://arxiv.org/pdf/2607.06618)): mIoU ranking, winner (Amazon) 0.391 mIoU; fine-tuned LLM given timestamped SRT and asked for the span; subtitle-only oracle setting included.
- **NMSQA / Spoken SQuAD** ([DUAL](https://arxiv.org/pdf/2203.04911)): AOS = temporal IoU of audio spans; the standard is ASR + text span QA.
- **[TimeStampEval](https://arxiv.org/abs/2511.11594)**: LLMs asked for raw timestamps score 37%; RapidFuzz prefilter + LLM verification on short snippets >90%; off-by-one boundary errors dominate. **[Constrained evidence selection](https://arxiv.org/html/2606.20890)**: have the LLM pick a chunk ID, never generate a number, to kill "temporal hallucination". [LongAudio-RAG](https://arxiv.org/html/2602.14612v4) does the same with timestamped event tables.
- **Medical data**: [PriMock57](https://github.com/babylonhealth/primock57) (57 consultations, 8 h 38 m, audio + utterance transcripts + notes) is the only free, close-fit dev set; [ACI-Bench/MTS-Dialog](https://github.com/microsoft/clinical_visit_note_summarization_corpus) are text only; [MeDial-Speech](https://arxiv.org/html/2605.26747v1) (2026, 111 h, CC-BY non-commercial) has no evidence timestamps. **No** public benchmark has exactly this shape (yes/no + supporting span + numeric hard negatives); the closest competitions are the MedVidQA family, and every winner was transcript-first.

## Recommendation

1. **Cascade, not audio-native, for the main path.** Qwen3-ASR-1.7B (or Whisper-large-v3/Voxtral as a second vote), Qwen3-ForcedAligner word timestamps, then an LLM that sees numbered, timestamped utterances and, for all ten questions, returns yes/no plus **utterance/word IDs**, never seconds. Convert IDs to seconds in code. Budget: ASR+alignment ~5 s, LLM ~10-20 s on one GPU.
2. **Audio-native only as a verifier, and only audio-in.** For questions where the answer hinges on a number/dose/drug and the two ASR votes disagree (or the LLM is low-confidence), crop the candidate span plus 1-2 s and ask Voxtral-Mini-3B (9.5 GB) or MOSS-Audio-4B a single closed question with **no transcript in the prompt**. Do not use any audio LLM for localisation.
3. **Tune boundaries on the training set (and PriMock57 if more is needed)**: IoU is 60% of the score, so calibrate span padding and utterance-vs-clause granularity empirically.

## Pitfalls

- WhisperX drops or misplaces timestamps on digits and units ("500mg", "2.5") ([#869](https://github.com/m-bain/whisperX/issues/869), [#1298](https://github.com/m-bain/whisperX/issues/1298)); spell numbers out before alignment or use Qwen3-ForcedAligner.
- ASR normalisation ("fifty" vs "50", "mg" vs "milligrams") breaks string matching against the question; normalise both sides.
- Gemma's 30 s window, Qwen3-Omni's VRAM, AF3/AF-Next non-commercial license, Ultravox v0.7's 355B backbone.
- Off-by-one span boundaries (TimeStampEval) and LLMs inventing seconds (constrained-selection paper).
- Unverified: MOSS-Audio English WER, open Voxtral Realtime timestamp output, FireRedAudio VRAM; check cards before committing.
