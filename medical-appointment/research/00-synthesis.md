# Medical appointment: what we measured, what the committee found, what to build

Written 2026-09-17, the day the starter kit landed. Reports 01 to 05 in this folder are the five committee members' full findings with sources. This page is the short version plus our own measurements on the 39 training conversations.

## The task in one paragraph

One request carries one recorded consultation as MP3 and ten yes/no questions. We return ten booleans and, for every yes, the start and end second of the passage that makes it true. The score is 0.4 times accuracy plus 0.6 times mean temporal intersection over union (tIoU, overlap divided by the union of the two spans) over every annotated yes question. A yes we answer no scores zero on both halves. We have 60 seconds per conversation on our own hardware and no cloud calls in the request path. The evaluation set is 38 conversations and we get one attempt.

## What the training data says

Measured with `span_ceiling.py` and a small script over `data/question_train.csv`.

**Spans are short and sit on sentence boundaries.** The 195 annotated spans have a median length of 2.9 seconds, an interquartile range of 1.9 to 4.0 seconds, and a maximum of 14.2 seconds. Ten of them are shorter than 1.3 seconds, and those cover a single phrase inside a sentence, not a whole sentence.

**Edge precision is the score.** If both edges of a predicted span are half a second off, mean tIoU across the training spans is 0.66. One second off gives 0.45. Padding a span by one second on each side to be safe costs the same as being a second off. Returning a whole turn or the whole clip is worth almost nothing, which the brief already says.

**The audio is clean and the transcript is close to perfect.** The conversations are simulated, read by two clear voices with no overlap and no background noise. faster-whisper large-v3 transcribed a 106-second clip in 10 seconds on the laptop's RTX 5070 and made no visible word errors on the sample checked. Word error rate is therefore a smaller worry here than on last year's real consultation audio.

**The annotations were made with a Whisper-family tool, and Whisper's word starts are early.** Every one of the 390 annotated timestamps is a multiple of 0.02 seconds, which is Whisper's token grid and also the frame stride of the wav2vec2 aligner that WhisperX (a wrapper that re-times Whisper output with a separate alignment model) uses. When we take faster-whisper large-v3 word timestamps, cut them into sentences at punctuation, allow merges of up to four consecutive sentences, and pick the best candidate for each gold span, the mean tIoU is 0.753. That number is a ceiling for a perfect selector at sentence granularity. The gold start is later than our start by a median of 0.36 seconds and the gold end is later than our end by a median of 0.12 seconds. Shifting every candidate by those two constants lifts the ceiling to 0.824. The offsets are consistent across conversations (interquartile range 0.24 to 0.48 seconds on starts), so this is a property of Whisper's timestamp method, not noise.

**Hard negatives are mostly semantic, not numeric.** Only 28 of 390 questions contain a digit, and only 10 of the 142 hard negatives do. The typical hard negative is "Should the asthma medication dose be increased?" against a conversation that says "no changes", or "Has an irregular heart rhythm been found?" against "heart sounds normal". A system that scores topical overlap answers yes to all of these. Twenty-one questions are tag questions ("..., correct?", "..., didn't it?").

## What the committee found

Five agents researched in parallel with web access and were told to verify dates, licences and numbers. The one-line verdicts:

1. **Speech recognition (report 01).** NVIDIA's Parakeet-TDT-0.6B-v2 (TDT is a token-and-duration transducer, a decoder that emits each word together with how long it lasted) is the recommended primary automatic speech recognition (ASR) model: native word timestamps from the same pass, a permissive CC-BY-4.0 licence, under a second per clip on any 24 GB GPU, no hallucination on silence. Its timestamps are on an 80 ms grid. The Whisper family has worse conversational word error rate and timestamps that drift by 100 to 400 ms. The 2026 models with the best text accuracy (MOSS-Transcribe, Cohere-transcribe, ARK-ASR) ship no timestamps at all.
2. **Diarization (report 02).** Skip it in the first version. A span is one utterance, not an exchange, so knowing who spoke does not localise evidence, and diarization systems' most common error is dropping exactly the short confirmations ("yes, okay") we need. If turn boundaries ever help, NVIDIA's streaming Sortformer v2.1 costs about a second per clip and has a permissive licence.
3. **Alignment and spans (report 03).** The 20 ms grid finding above is theirs. Their recommended aligner is ctc-forced-aligner or torchaudio's MMS_FA, both built on Meta's Massively Multilingual Speech (MMS) model, run on a copy of the transcript with digits spelled out, because every wav2vec2-style aligner drops timestamps on digits. Qwen3-ForcedAligner-0.6B (Apache-2.0, aligns digits) is the second opinion, with a known bug that gives about 2.5 percent of words zero duration. Units should be sentences or clauses split further at pauses over half a second. Winning entries in the two published competitions with this shape of metric, MedVidQA (medical instructional video answer localisation) and the NLPCC 2026 shared task, all anchored on transcript timestamps and then merged adjacent units with a length prior.
4. **The answering large language model (LLM, report 04).** Qwen3.6-27B or Qwen3.8-27B with thinking disabled, served by vLLM with prefix caching so the server encodes the transcript once and the ten questions decode in one batch. Render the transcript as numbered segments and have the model return segment identifiers, never seconds. Put a verbatim quote field before the answer field in the JSON schema so the model re-reads the detail before deciding. Rewrite tag questions to declaratives before prompting, since a 45-model study found tag suffixes shift yes/no answers by up to 32 points. Natural language inference cross-encoders are a sub-second fallback but are documented as weak on numeric near-misses.
5. **Audio-native models (report 05).** Do not put an audio LLM in the localisation path. The best open model on TAG-Bench, a 2026 benchmark that asks audio models to locate a spoken event in time, reaches a mean intersection over union of 31 percent. Audio LLMs given both audio and a transcript side with the transcript. The only sensible role is a verifier for numeric questions, fed a cropped span with no transcript in the prompt.

## The pipeline this points to

```
MP3 -> decode once to 16 kHz mono
    -> ASR with word timestamps (Parakeet-TDT-0.6B-v2, or faster-whisper large-v3 as we have now)
    -> optional forced alignment on a digits-spelled-out copy (MMS aligner) for tighter edges
    -> sentence/clause units with start/end from first/last word, plus the fitted edge offsets
    -> numbered transcript into a local LLM (Qwen3.x-27B, vLLM, prefix cache, JSON schema)
    -> one request per question in parallel: {quote, answer, segment ids}
    -> ids back to seconds, merge cited neighbours only when both are needed
    -> validate_response, and on any exception return a guess with no span
```

Budget on one 24 GB GPU: ASR one to ten seconds depending on model, alignment about one second, LLM two to ten seconds for all ten questions. That leaves most of the 60 seconds unused, which is where a second ASR vote or an audio verifier could go later.

## First experiments, in order

1. **Edge offsets by tool.** Run the 39 files through faster-whisper native, WhisperX, MMS_FA and Parakeet-TDT, and for every gold edge record the signed offset to the nearest word boundary. The tool whose offsets cluster at zero on the 20 ms grid is the one the annotators used. `span_ceiling.py` already computes this for faster-whisper. This is the cheapest experiment with the largest expected gain on the 0.6 half of the score.
2. **Clause units.** Add a split at pauses over 0.5 seconds and at commas, and see whether the ten sub-1.3-second spans become reachable without hurting the rest.
3. **Answering baseline.** Wire any local instruction model over the numbered transcript, score on the 390 questions with the evaluator, and read the hard-negative row. That row is the number to move.
4. **Serving.** Get the endpoint reachable from the internet on a cloud virtual machine this week, with a request size limit above 5 MB and the model warmed at import. The brief's warning that one dead request costs ten marks is the one to take literally.

## Open questions

- Whether the offsets differ between the doctor's and the patient's voice, or between sentence-initial and mid-turn words. A per-position offset table would follow from experiment 1.
- Whether Parakeet's 80 ms grid costs more in edge precision than it gains in word error rate on this clean audio. Report 01 and report 03 disagree on which to lead with; the offset experiment settles it.
- Whether the evaluation conversations are the same voices and the same recording setup as the training ones. Nothing in the brief says so.

## Files

- `transcribe_cache.py` writes `transcripts/<file>.<model>.json` with segment and word timestamps. The folder is gitignored.
- `span_ceiling.py` reports the oracle-selection ceiling, the edge offsets and the worst spans for a cached model.
- `research/01-05` are the committee reports with sources and uncertainty flags.
