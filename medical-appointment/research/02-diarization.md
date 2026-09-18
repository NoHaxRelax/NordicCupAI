# Committee report 2: diarization and speaker-attributed ASR (verified 2026-09-17)

**Short answer:** for this task diarization is mostly wasted budget. Skip it in v1. If you add it later, use it only as a cheap turn-boundary hint (Sortformer v2.1 or pyannote community-1, ~1-4 s per clip), never as a hard dependency.

## Why: what the training data says

Measured on the 195 positive evidence spans in `data/question_train.csv` (39 clips, 74-232 s, median 106 s):

- span length: median **2.9 s**, p75 4.1 s, p90 5.5 s, max 14.2 s; **85% <= 5 s, 99% <= 10 s**
- i.e. a span is one utterance ("100 mg once a day for two weeks"), not a doctor/patient exchange.

So "who said it" does not localize evidence; precise word/sentence timestamps do. The 0.6 tIoU weight rewards tight utterance boundaries, which come from ASR word timestamps plus pause/turn segmentation. Speaker labels would help only on the rare correction pattern (patient: "so 200 mg?", doctor: "no, 100"), which an LLM already resolves from sequence. Diarization can also *hurt*: all systems' dominant error is **missed short speech** (backchannels like "yes, okay", exactly the confirmation utterances) and 2-speaker DER of ~10% means one in ten seconds is mislabeled ([Benchmarking Diarization Models, arXiv 2509.26177](https://arxiv.org/abs/2509.26177)). A 2025 clinical study also found lexical cues alone identify doctor vs patient roles ([medRxiv, Speaker Role Identification in Clinical Conversations](https://www.medrxiv.org/content/10.1101/2025.08.14.25332837v1)), so if you want roles, ask the LLM.

## Comparison table

RTF = audio-seconds per compute-second on RTX A6000 (ETH paper) unless noted; "3-min est." is derived from RTF. DER collar noted per source.

| System | License / access | 2-spk DER (CALLHOME unless noted) | 3-min est. | Word to speaker | Overlap |
|---|---|---|---|---|---|
| [pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | MIT, **HF-gated** | 28.5% CALLHOME (0 collar, own card); 19.9% 2-spk agg. (0.25 s, ETH) | ~4 s (RTF 45x) | WhisperX `assign_word_speakers` (max-overlap lookup) | multi-label; STT still drops overlapped words |
| [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (pyannote.audio 4.0, late 2025) | CC-BY-4.0, **HF-gated**; runs offline from local clone | 26.7% CALLHOME, 17.0% AMI-IHM, 20.2% DIHARD3 (0 collar) | not published; likely ~3-5 s (unverified) | built-in `exclusive_speaker_diarization`, one speaker per instant | exclusive mode assigns overlap to dominant speaker |
| pyannoteAI precision-2 | **paid API**, closed | 16.6% CALLHOME, 12.9% AMI-IHM | n/a | API | violates "no cloud" |
| [Sortformer v1 offline](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | **CC-BY-NC-4.0** | 5.85% CALLHOME-2spk (0.25 s); 14.76% DIHARD3 | ~1 s (RTF 165x) | NeMo timestamps to ASR masking | end-to-end, <=4 spk; higher confusion than others (ETH) |
| [Streaming Sortformer v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | CC-BY-4.0 | 6.57% CALLHOME-2spk (0.25 s); 13.24% DIHARD3 <=4spk | ~1 s (RTF 210x; card: RTF 0.002 at 30 s-latency config) | same | <=4 spk; use 30.4 s "very high latency" config = offline |
| [Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | NVIDIA Open Model License | 6.65% CALLHOME-2spk; 16.67% AMI-IHM (1 s latency) | ~1 s | same | same |
| NeMo MSDD (TitaNet+MSDD, telephonic) | NeMo (Apache-2.0); NGC | legacy cascaded; CH109-tuned; no current numbers found | few s | NeMo | weakest; superseded by Sortformer |
| [DiariZen large-s80-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) | code MIT, **weights CC-BY-NC** | 11.4% 2-spk agg. (ETH); 13.9% AMI-SDM, 14.5% DIHARD3 (0 collar) | ~9 s (RTF 20x, slowest) | pyannote-style pipeline | best open DER, slowest |
| [WhisperX](https://github.com/m-bain/whisperX) | BSD-2; now wraps community-1 | = pyannote | ASR + ~4 s | interval lookup per word | README: "overlapping speech is not handled particularly well" |
| [multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) | NVIDIA Open Model License | cpWER 15.8% CH109, 21.3% AMI-IHM | unknown; one ASR instance per speaker + Sortformer | native speaker-tagged SegLST | handles full overlap (per-speaker instances) |
| [MOSS-Transcribe-Diarize 0.9B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (2026-07) | Apache-2.0 | no DER/cpWER published; only CER/cpCER (Podcast 6.0/7.4) | RTF 0.06-0.12 on H100, ~11-22 s | end-to-end `[start][Sxx]text[end]`, **segment-level only** | unknown |
| [Voxtral Transcribe 2](https://mistral.ai/news/voxtral-transcribe-2/) | Realtime = Apache-2.0 (no diarization); Mini Transcribe V2 with diarization = **API-only** | - | - | - | - |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (2026-01) | Apache-2.0 | no speaker labels; timestamps via separate ForcedAligner-0.6B | - | - | - |
| Meta Muse Voice Transcribe (2026-09-01) | **API-only**, closed | 17.5% avg AMI/VoxConverse claim | - | - | - |
| Kyutai STT / Unmute | CC-BY-4.0 | no diarization component found (uncertain) | - | - | - |
| 3D-Speaker (ModelScope) | toolkit; license unverified | only in-house DER (5.4%) | - | - | Chinese-centric |

Sources: ETH RTF/2-spk table ([arXiv 2509.26177](https://arxiv.org/abs/2509.26177)); Nemotron 3.5 multispeaker PR still open ([NVIDIA-NeMo/Speech #16277](https://github.com/NVIDIA-NeMo/Speech/pull/16277)); failure-mode survey ([Kili, Aug 2026](https://kili-technology.com/blog/speaker-diarization-models-guide-benchmarks-and-failure-modes-2026)).

## Recommendation

1. **v1: no diarization.** Spend the budget on ASR with reliable word timestamps and on a segmenter that cuts at pauses (VAD) so candidate spans match the 2-5 s utterance shape. Let the LLM infer doctor/patient from wording.
2. **v2 (optional, <=2 s):** Streaming Sortformer v2.1 at the 30.4 s-latency config (offline, permissive license, no HF gating, ~1 s/clip) as a *turn-boundary* signal: speaker changes are strong candidate span edges, and turn labels let the LLM prefer the doctor's statement over the patient's guess in correction dialogues. Alternative: pyannote community-1 with `exclusive_speaker_diarization` if you already carry pyannote (WhisperX). Word to speaker: assign each word to the diarization segment with max overlap; fall back to the utterance's majority speaker.
3. **Do not** use end-to-end speaker-attributed ASR (MOSS, multitalker Parakeet) here: MOSS gives segment-level timestamps only and costs 11-22 s; multitalker Parakeet doubles ASR cost and is optimized for overlap you probably don't have (simulated dialogue: check a few clips; if TTS-generated, overlap is ~0 and voices are trivially separable).

## Pitfalls

- pyannote models are HF-gated: accept terms, download once with a token, `git lfs` clone, load from local path at server start. Nothing in the request path touches HF.
- NeMo is a heavy dependency; co-installing NeMo + WhisperX/pyannote in one env commonly breaks on torch/transformers pins. Pin versions early.
- Licenses: Sortformer **v1** and DiariZen weights are non-commercial; v2 (CC-BY-4.0) / v2.1 (NVIDIA OML) are the safe Sortformer picks.
- All these models expect 16 kHz mono; resample the 44.1 kHz MP3.
- DER on CALLHOME is quoted with a 0.25 s collar by NVIDIA and 0 collar by pyannote; not directly comparable.
- Diarization boundary error (~350 ms typical) is the same order as the tIoU margin on a 3 s span; don't let diarization edges override ASR word times.

**Uncertainties:** community-1 inference speed is unpublished (pyannote's "31 s/hour" claim appears only in secondary sources, hardware unknown); MOSS English DER/cpWER not published; Sortformer v2.1 release date not stated; Kyutai's lack of diarization is inferred from absence, not confirmed; 3D-Speaker license unverified.
