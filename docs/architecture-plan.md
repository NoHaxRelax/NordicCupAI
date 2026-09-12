> Produced by a 51-agent research fan-out (4.5M tokens, 1,037 tool calls) on 12 Sep 2026,
> then checked by hand against our own data. Corrections applied after that check:
>
> - **Verified true:** the top-20 worst words carry 563 of 1136 combined sub+del errors (49.6%);
>   `wer_no_backchannel` is a 5.3% relative reduction; `conditional` really is 28 true / 29 false;
>   the follow-up ceiling really is (16+35)/57 = 0.895.
> - **Verified true and I had missed the mechanism:** both Ambolt speech forks were *archived*
>   on 2026-08-31, which is exactly the metadata-only change I could not account for.
> - **WRONG, do not act on:** "a 9.3-minute consultation is 5 to 6 MB base64". Base64 inflates
>   by 4/3, so 17.9 MB raw becomes 23.8 MB. The recommendation to raise the body limit stands;
>   the number does not.
> - **WRONG, do not act on:** "Ambolt added a Danish csm-1b executor to speaches". Both forks
>   carry zero Ambolt commits and are identical to or behind upstream. Whatever csm-1b support
>   exists is upstream code. The forks are still real signal; the attribution is not.
> - **Not covered:** the clinical NER and ontology-linking dimension (scispaCy, MedCAT, UMLS,
>   SNOMED) died on a schema retry cap and never ran. Treat that area as unresearched.
> - **On the 32-of-35 "refuted" rate:** the verifiers were told to default to refuted under
>   uncertainty and to reject claims whose design implication overreached. Reading them, most
>   say the facts check out and the *conclusion* was too broad. It is not a 91% error rate.

# Nordic AI Cup 2026, Challenge 3: recommended architecture and verdict on your proposal

## 0. The short answer

Your pipeline is **right at both ends and wrong in the middle**.

- **ASR in, small local LLM with a heavily engineered prompt, small structured JSON out.** Keep all of it. That is the correct spine and the correct output prior.
- **Chunk, embed, retrieve-by-keyword-set, expand ±5.** Delete it. On your own corpus it is arithmetically a no-op that can only lose evidence.
- **The keyword sets and the embedding model survive**, repointed: the keyword sets become a high-recall lexical sweep that annotates and never discards, and the embedding model points at the label space and at example selection rather than at transcript chunks.

The thing your design is missing is not retrieval. It is a **second recall channel**: an explicit per-candidate checklist pass. Your dominant failure mode is omission, and a single generative pass has no mechanism that prevents forgetting.

---

## 1. Four measurements from your own repo that change the design

I re-read `labels/primock57_structured.json`, `docs/results/asr_baseline.json`, `docs/results/structured_baselines.json` and `docs/results/parakeet_bench.json` before designing. Four numbers in the brief and in the panel material are wrong, and each one moves a decision.

### 1.1 The label space is 57 distinct conditions, not 15 to 20

After running your own `medvoice.structured.canon` over the 57 gold label sets:

| | value |
|---|---|
| consultations | 57 |
| diagnosis mentions | 92 (70 working, 22 rule-out) |
| distinct canonical labels | 57 (60 raw surface forms) |
| labels occurring exactly once | 43 |
| top-20 labels cover | 59.8% of all mentions, 68.6% of working mentions |
| cardinality | 32 singles, 17 pairs, 7 triples, 1 with five |
| head | gastroenteritis 6, migraine 6, uti 6, urti 5, asthma exacerbation 4 |

**Consequence:** a fixed enum built from PriMock57 caps recall near 60% of mentions before inference begins, and the held-out set will contain conditions you have never seen. Do not hard-constrain the diagnosis field to a closed enum. Generate open free strings, canonicalise in Python where a miss is recoverable, and use a candidate list as a *question list* rather than as a decode constraint.

This is the single biggest correction. The whole "closed-set classification" framing that several strands of the research lean on does not hold at this label cardinality.

### 1.2 Per-channel decoding is worse on deletions too, not just insertions

The panel's theory was that per-channel decoding loses to mixed only because each track hallucinates into the other speaker's silence, and that VAD-gating would fix it. Your data refutes that:

| large-v3 | WER | S | D | I |
|---|---|---|---|---|
| mixed | 0.1385 | 446 | 690 | 213 |
| per-channel | 0.3945 | 1542 | 1308 | 992 |

Deletions nearly double and substitutions more than triple. Insertions alone would be a silence-hallucination signature; this is broad degradation. **Mix down for recognition.** Use channel identity, when present, only for speaker attribution. Do not spend a morning on a VAD-gated per-channel merge.

### 1.3 "Deletions dominate" is true but it is mostly backchannel, and the 82% figure circulating in the panel is wrong

`error_profile()` in `medvoice/metrics.py` counts ops `S` **and** `D` together, so `worst_words` is not a deletion list. For large-v3 mixed, the top-20 worst reference words sum to 563 out of 1136 combined S+D errors, which is **49.6% of the substitution-plus-deletion mass concentrated on 20 token types**, every one of which is a function word or backchannel: ok, i, is, you, yeah, and, it, that, the, a, no, have, to, am, are, your, like, bye, ohh, or.

Corroborating: `wer_no_backchannel` for the same run is 0.1311 against 0.1385, a 5.3% relative reduction.

**Consequence:** the deletion-dominated error profile is real but you have no evidence yet that it touches *scored content*. Do not adopt the aggressive recall-biased decode settings (VAD off, `no_speech_threshold` 0.8 to 0.9, `log_prob_threshold` -1.5) that three of the four designs proposed. Those buy back "ok" and "yeah" while raising hallucination risk, and a hallucinated condition costs you more than a missing "yeah". Run the content-term recall measurement on day 1 and let it decide (see §8.1).

### 1.4 The follow-up ceilings are hard and low

From your oracle row: a follow-up cue is present in the transcript for **44 of 57** consultations, and the interval is actually spoken in **35 of the 41** consultations that have one. Sixteen of 57 gold intervals are null.

Best achievable interval accuracy, assuming you get every null right: (16 + 35) / 57 = **0.895**. Best achievable follow-up-cue recall: **0.772**. About a quarter of gold follow-up information was never said out loud and is unrecoverable from audio at any ASR quality.

Stop optimising when you touch those. Also note the baselines you have to beat:

| | majority | prior | lexical |
|---|---|---|---|
| dx_f1 | 0.0 | 0.0965 | 0.0 |
| urgency_exact | 0.4561 | 0.4561 | 0.2632 |
| urgency_undercall | 0.2807 | 0.2807 | **0.0** |
| interval | 0.2807 | 0.2807 | **0.4386** |
| modality_exact | 0.4386 | 0.4386 | 0.4386 |
| conditional_exact | 0.4912 | 0.4912 | 0.4912 |

Two things fall out. First, **the lexical baseline already beats the majority baseline on interval by 15.8 points**, so the deterministic regex normaliser is justified before any model runs. Second, **`conditional` is a coin flip at 0.4912 for every baseline**, and the label file's own schema note ("almost always true") contradicts its own data (29 False, 28 True). Populate that field because an omitted field scores zero, but spend no prompt budget and no A/B slot on it.

---

## 2. Recommended architecture: Spine + Checklist

One FastAPI process, one GPU, five stages, nothing crossing a network boundary at inference. This is Design 1's straight-through spine with the checklist channel from Design 3, the never-delete record discipline from Design 2, and the latency tier ladder from Design 4.

```
audio ──► [0] INGEST ADAPTER  (5 transports behind one signature)
       ──► [1] ASR, mixed channel, tier-selected  ──► NUMBERED TRANSCRIPT  ◄── first-class output
       ──► [2] LEXICAL SWEEP  (annotates, never discards)
       ──► [3a] GENERATIVE PASS   (full transcript, free text, no grammar)
           [3b] CHECKLIST PASS    (one yes/no per candidate label, cached prefix)
                        └── union ──► [3c] PROJECT PASS (grammar-constrained, ~250 tok)
       ──► [4] DETERMINISTIC POST  (interval regex, emergency floor, canon, snap, validate)
       ──► [5] DTO ADAPTER  ──► organiser's response model
```

### [0] Ingest adapter, ~60 lines, written before the event

`load_audio(request) -> (float32 mono 16 kHz, meta)`. Five branches tried in order: base64 under any of `audio | audio_base64 | wav | data | file | recording`; an http(s) URL to fetch; a multipart upload; a session-keyed chunk-stream accumulator; and a plain transcript string (skip ASR entirely). Sniff with `soundfile`, fall back to ffmpeg. Raise the uvicorn body limit to 32 MB, because a 9.3-minute 16 kHz consultation is roughly 18 MB raw and 5 to 6 MB base64, an order of magnitude above any image payload in Ambolt's history.

Five editions, sixteen use cases, zero audio. There is no transport precedent. This adapter is the thing that stops a transport surprise from costing half of day 1.

### [1] ASR, mixed channel, tier ladder as a config dict

Not an architectural commitment. A dictionary of decode kwargs chosen on day 1 once the real per-item budget is known. Populate it from your own measured real-time factors against a 558-second median consultation:

| tier | model | WER | RTF | time / consultation |
|---|---|---|---|---|
| 0, budget > 60 s | large-v3, beam 5 | 0.1385 | 17.4 | ~32 s |
| 1, budget 10 to 60 s | distil-large-v3 | 0.1545 | 58.3 | ~9.6 s |
| 2, budget 3 to 10 s | large-v3 batched (VAD chunking, batch 16) | ~0.14 | measure | ~3 to 5 s |
| 3, budget < 3 s | ASR only, deterministic extraction, no LLM | | | |

Note that `distil-large-v3` dominates `small.en` (0.1648, RTF 38.5) and `base.en` on **both** axes at once, so the ladder skips them. Keep `parakeet-ctc-0.6b` (0.1423 on CPU at 9.4x realtime, roughly 59 s) as the no-GPU escape only.

Settings that are defensible on your evidence: `condition_on_previous_text=False` (kills the repetition collapse that swallows whole segments in overlapping dialogue), VAD on with padded boundaries (`speech_pad_ms=400`, `min_silence_duration_ms=500`), `beam_size=5`, `word_timestamps=True`, and an `initial_prompt` seeded with the 60-entry alias table already in `medvoice/structured.py` plus common drug names.

Output format, and the line numbers are load-bearing:

```
L14 [08:31] CLINICIAN: ...text...
L15 [08:39] PATIENT: ...text...
```

Any provenance, evidence, timestamp or turn-index field the organiser invents is then a lookup rather than a re-run.

### [2] Lexical sweep, ~40 trigger terms

Regex over the full transcript, tuned deliberately to over-trigger. Three jobs, none of them filtering:

1. Mark lines carrying follow-up or safety-netting language ("come back", "see you", "follow up", "book", "pop back", "give me a shout", "keeping an eye on it", "if it doesn't settle", "any problems", weekday names, "in X days/weeks/months").
2. Pull candidate time phrases for the normaliser. This is what buys the 0.4386 interval baseline.
3. Raise the emergency flag ("ambulance", "999", "blue light"). Your oracle says this is 5 of 5.

Nothing is removed from the transcript. The sweep appends a short block at the **end** of the prompt, after the transcript, because recency helps: `candidate follow-up lines: 14, 189, 203`.

### [3] Extraction: one transcript, two recall channels, one projection

Median 2,050 tokens, maximum 3,780. That is 1.5% of a 256k window. The whole transcript goes in uncut.

**[3a] Generative pass, unconstrained, temperature 0.** "List every condition named or implied by either speaker, with the quoted line and whether it was asserted, denied, raised to exclude, or voiced by the patient as a worry. List every utterance about coming back, with its quoted line." No grammar here on purpose: the measured format tax is about 3.9 points from the format-requesting prompt and only 1.6 more from the decoder, and a grammar admitting only the final answer removes the intermediate states the model needs to compute.

**[3b] Checklist pass, the graft that matters.** One constrained yes/no per candidate label over a vLLM-cached transcript prefix; read the yes-probability and threshold on dev. A label cannot be forgotten if it is explicitly asked about. This is the only mechanism in any of the four designs that attacks omission at its source.

Candidate list construction, and this is where your embedding model goes: take the union of (a) the top ~25 conditions by frequency in whatever label inventory exists, and (b) 20 to 30 retrieved by cosine similarity between the transcript and the condition vocabulary. Because 43 of 57 labels are singletons, the shortlist is a **prior, never a filter**: pass [3a]'s free-string candidates through even when they are not on the list.

Cost check before you build it: 45 to 55 yes/no calls at ~5 output tokens each over a cached 2,050-token prefill. On a warm vLLM server that is sub-second batched. If the revealed per-item budget is tight, this is the first thing to shrink (drop to the top 20 retrieved), and the second thing to cut entirely.

**[3c] Projection pass, grammar-constrained, ~250 tokens.** Reads [3a] and [3b] output only, never the transcript again. vLLM `guided_json` with the xgrammar backend, one schema compiled once at server start. Never a per-request schema: dynamic enums have been measured at 60 s of compile at 500 members.

Schema, and **field order is load-bearing** because JSON key order under a grammar is generation order:

```json
{
  "evidence": {"dx_lines": [], "dx_quote": "", "followup_lines": [], "followup_quote": ""},
  "conditions": [
    {"label": "", "rank": 1, "status": "working|ruled_out|historical|family|patient_concern", "line": 0}
  ],
  "followup": {
    "urgency": "emergency|urgent_today|days|weeks|conditional_only|none",
    "interval_phrase": "the spoken words, verbatim, or null",
    "modality": "ambulance|a_and_e|f2f_gp|remote_gp|blood_test|specialist|physio|self_referral|pharmacy|none",
    "booked": false, "conditional": true, "trigger": ""
  }
}
```

Two invariants carried from Design 2, both cheap and both required by your own gold schema:

- **Never delete, only mark.** `diagnoses_to_exclude` is a first-class gold field (18 of 57 consultations have one, 22 of 92 mentions) and `patient_concerns` exists separately. A ruled-out condition stays in the array with `status: ruled_out`; the projection decides whether to emit it. That turns a precision-versus-recall schema surprise into a one-line filter instead of a pipeline re-run.
- **Keep the surface form beside every normalised value.** `interval_phrase` next to the computed `[min, max]`. A span-scored metric and an exact-match metric are then both one field access away.

Constrained decoding applies to the four genuinely closed fields (`urgency`, `modality`, `booked`, `conditional`) and to nothing else. `label` is an open free string.

### [4] Deterministic post-processor, ~150 lines of pure Python, unit-testable without a GPU

1. `parse_interval(phrase) -> [min,max] | None`. About 50 lines of regex over a small closed vocabulary. "three to five days" to [3,5], "a couple of weeks" to [14,14], "48 hours" to [2,2], "straight away" to [0,0], unrecognised to None. Ranges, not points, because your `interval_score` already scores range overlap. **Null is a first-class, rewarded answer**: 16 of 57 gold intervals are null, so a system that always emits a number fails 28% of the set while looking fine on average. The LLM never does calendar arithmetic; published generative baselines run a 21 to 24 day mean absolute error on exactly this against roughly zero for a rule grammar.
2. `canon(term)`, already in the repo.
3. **Emergency floor.** If the lexical sweep fired, force `urgency=emergency` and `modality=ambulance` regardless of the model. Twelve lines of regex defending your most expensive error class, oracle 5 of 5. Note the narrow version only: the *wide* lexical urgency rule trades `urgency_exact` from 0.4561 down to 0.2632 to buy undercall down to zero, which is a loss under any plain exact-match metric and a win only under your own asymmetric `urgency_cost`.
4. Validate and repair: `jsonschema.validate`, then snap out-of-enum strings to the nearest legal value with `difflib`/`rapidfuzz`, then one retry at temperature 0. Budget 1.2 calls per request.

### [5] DTO adapter and serving

Twenty hand-written lines mapping the internal record to whatever `models/dtos.py` says. **Write it by hand after reading the DTO.** Do not build reflective schema binding with fuzzy field-name matching: its failure mode is silent binding to the wrong field, and it would sit on the critical path at 10:30 on day 1.

Serving, cloned from the shape already working in `serve/api.py`: `POST /predict` with the organiser's response model copied verbatim, `GET /` returning "Your endpoint is running!", `GET /api` returning service name and uptime from a module-level `start_time`. Plus three things `serve/api.py` lacks:

- a FastAPI **lifespan handler** that loads every model and runs one synthetic warm-up (5 seconds of silence, one dummy prompt) before the app yields, with `/api` returning 503 until warm;
- an `asyncio.wait_for` wrapper at **80% of the published budget** whose `except` branch returns a schema-valid degraded object (a wrong answer scores more than a hang in every harness in this family);
- self-validation of our own output before the response leaves.

Temperature 0, fixed seed, pinned model revisions, every raw request and response body appended to `runs/event/requests.jsonl`.

### Models

| role | pick | note |
|---|---|---|
| ASR tier 0 | `Systran/faster-whisper-large-v3` | measured 0.1385 / RTF 17.4 |
| ASR tier 1 | `distil-whisper/distil-large-v3` | 0.1545 / RTF 58.3, dominates small.en |
| ASR CPU escape | `nvidia/parakeet-ctc-0.6b` | already on disk, 2.3 GB |
| Danish | `syvai/hviske-v3` (Whisper-flavoured), `alexandrainst/roest-315m` as backup | download now, do not tune |
| Extractor primary | `Qwen/Qwen3-14B` (AWQ or FP8 on a 24 GB L4) | |
| Extractor fallback | `Qwen/Qwen3-8B` | same family, tokenizer, template, grammar: a config change |
| Quality arm, 48 GB only | `google/gemma-3-27b-it` | |
| Embeddings | `BAAI/bge-m3` | label shortlist and example selection only |
| Excluded, argue down in one sentence | MedGemma 27B, OpenBioLLM, BioMistral, Meditron, Med42, JSL-Med (CC-BY-NC-ND), Ministral-8B (research licence) | medical adaptation beats its own base in 12.1% of cases, ties 49.8%, loses 38.2% once prompts are optimised per model |

Thinking mode **off** on the extraction call. Set `--structured-outputs-config.enable_in_reasoning=True` if you use a thinking-capable checkpoint at all, or the grammar silently stops applying.

---

## 3. Your proposal, component by component

| your component | verdict | why |
|---|---|---|
| ASR front end | **keep** | correct, and the cascade still beats audio-native models on speech understanding |
| two-channel input | **mix down** | your own table: per-channel 0.3945 vs mixed 0.1385, worse on S, D and I |
| chunk transcript small | **drop** | 1,523 words at 100-word chunks is 15 chunks |
| embed chunks | **drop** | embeddings over 13.9% WER text; structured retrieval amplifies upstream ASR error 36 to 67% relative to naive dense retrieval, and 87 to 96% of the damage lands on named entities, which is exactly the disease names |
| retrieve top-k by keyword-set similarity | **drop as a retriever** | "disease mentions" is a diffuse category query against a transcript where every chunk is about symptoms; cosine similarity is near-uniform and carries almost no ranking signal |
| expand ±5 neighbours | **drop here, move to ASR windowing** | one hit expanded ±5 spans 11 of 15 chunks, 73% of the median transcript; two hits in different halves returns 100% |
| predefined keyword sets | **keep, demoted** | as a lexical annotator they already beat the majority baseline on interval by 15.8 points and catch 5 of 5 emergencies |
| the embedding model | **keep, repointed** | at the label space and at example selection |
| small local LLM | **keep** | and it is forced anyway: the closest healthcare precedent caps VRAM at 24 GB and forbids cloud APIs at inference |
| heavily engineered prompt | **keep, upgraded** | constrained decoding on the closed fields, two-stage, evidence-first key order |
| small structured JSON out | **keep** | strongest prior in your whole proposal: 16 of 16 prior use cases returned a small typed object, never free text, never a text-similarity metric |

### The arithmetic that kills the retrieval stage

Median transcript 1,523 words. At 100-word chunks that is **15 chunks**, 28 at the corpus maximum. One retrieved chunk expanded ±5 spans **11 chunks, 73% of the median document**. Two keyword sets at top-5 gives up to 10 hits; any two landing in different halves returns everything.

Token budget before retrieval: ~2,050. Token budget after: ~2,050.

It buys no compression, no latency (ASR dominates the critical path by two orders of magnitude), and no context headroom. It is a lossy identity function. And it adds an 8 to 12% omission-only stage (best measured within-document chunk recall at k=5, with the best commercial embedder) on top of a pipeline where omissions are already 64% of critical errors and ASR deletions already exceed substitutions at every model size.

You cannot keep both the retrieval and the expansion and claim a benefit from either.

### Where your instinct was right for a reason you did not state

**(a) You wanted less junk in front of a small model.** That intuition is correct and is supported: FLenQA measures accuracy falling from 0.92 at 250 tokens to 0.68 at 3,000 tokens, purely from irrelevant padding, and MedAlign measures GPT-4 losing 8.3% accuracy moving from 32k to 2k context. But the fix that follows is *remove junk*, not *retrieve a subset*. The version of your idea that works: strip backchannel, filler and repeated confirmation from the transcript before the prompt. That shortens a heavy-overlap GP dialogue materially without dropping any content-bearing span, and it costs about fifteen lines.

**(b) You wanted a fixed keyword set for diseases.** The underlying instinct, turning open generation into constrained selection, is right and well evidenced: on DDXPlus (pick 1 of 49 diseases) constrained JSON-mode improved Gemini by 43.3 points and Llama-3-8B by 10.6. What you could not know is that your own label space is 57 distinct conditions with 43 singletons, so the constraint has to be a *question list*, not a decode enum. Same instinct, one layer up. That is what the checklist channel is.

**(c) You expected a small typed JSON rather than a note.** This is the most valuable prediction in the whole proposal and it is well supported: five editions, sixteen use cases, every graded field an int, a float, a closed-vocabulary string, a list of those, or a base64 image. Never free text, never ROUGE. Your own ROUGE ceiling work (single-reference ROUGE-L 0.1782 for a clinician against another clinician, 0.3211 against the best of several references) is the reason *not* to bet on note generation, and you had already measured it.

**(d) You wanted a small local model.** Correct, and forced rather than chosen: 24 GB VRAM and fully offline inference in the nearest precedent, with ASR and the LLM co-resident.

**(e) The ±5 expansion instinct is right at the wrong layer.** Entities that straddle a hard 30-second ASR window boundary are genuinely lost. Overlapping windows with 5 seconds of context each side have been measured to lift PERSON entity F1 from 0.50 to 0.65 and cut numeric character error rate from 0.22 to 0.42 down to 0.08 to 0.18. Move the padding instinct there.

---

## 4. What to build before the event (12 to 16 September)

This block is the highest-leverage in the whole plan because it is free time that is not competing with the README. Everything here is schema-independent by construction.

**Friday 12 to Saturday 13**

1. **Pull every weight to the UCloud 50 GB team drive** and verify each loads with `HF_HUB_OFFLINE=1`. large-v3 CT2, distil-large-v3, the two Danish checkpoints, Qwen3-14B-AWQ, Qwen3-8B, bge-m3. Budget roughly 35 GB against 50 GB, so check before the event rather than during it. Accept any gated licences now (this is the one thing that cannot be done offline).
2. **Pin every dependency** into one `requirements.txt`, build the environment end to end, rebuild it once from scratch in a clean directory and confirm it still serves. Over half of submissions in a comparable served competition failed to build, mostly from unpinned dependencies and upstream breaking changes.
3. **Stand up the nginx public-link job** and get a green connection test against a stub `/predict` returning a hard-coded object. This is the failure mode the organisers warn about every single year.

**Sunday 14 to Monday 15**

4. **Ingest adapter**, all five transports, plus the fuzz suite: zero-length audio, a 25-minute recording, mono and stereo, 8 kHz, non-English, corrupted base64, ten concurrent requests. Every case must return a schema-valid object inside budget.
5. **Serving shell**: lifespan warm-up, 503-until-ready, `asyncio.wait_for` with a schema-valid degraded return, output self-validation, request logging.
6. **Deterministic post-processor**: `parse_interval` with its regex table, the emergency floor, `canon`, enum snapping, validate-and-repair. Unit tests, no GPU needed. Verify it reproduces the 0.4386 lexical interval baseline.
7. **ASR tier ladder** as a config dict, each rung loaded and smoke-tested on one file.

**Tuesday 16**

8. **Extend `medvoice/structured.py:score_all`** with per-field recall reported separately, the null branch scored as its own slice, and per-consultation win/loss/tie counts. Hold out 15 of the 57 as a never-tuned set.
9. **Build the content-term set** for the day-1 measurement: every canonical diagnosis token, drug name and interval phrase from the gold labels, ready to feed `medvoice.metrics.term_recall`.
10. **Danish rehearsal.** Run five PriMock57 transcripts through a Danish TTS model and measure the Danish ASR arm's WER on it. Two hours that converts the Danish risk from a lost day into a ten-minute branch. The signal is real: Ambolt forked the `speaches` server on 2026-02-10 and added a Danish `csm-1b` conversational-TTS executor with two speaker tokens, then archived both speech forks on 2026-08-31, seventeen days before the event.
11. **Write the two-stage prompt** against a placeholder schema, wire vLLM with xgrammar, confirm the evidence field generates first, and run it over all 57 consultations. You should already know your dev-set score before the event opens.

If items 1 to 3 slip, day 1 turns into a build day and the whole plan compresses. Do those three first.

---

## 5. The four event days

**Ownership: A = infra and portfolio, B = ASR, C = extraction, D = evaluation.** A is the one who can be pulled onto Challenges 1 and 2, which is why A owns nothing on the critical path after day 1 morning.

### Day 1, Thursday 17 September, 10:00 to 22:00

- **10:00 to 10:30, A alone.** Read the README, `models/dtos.py` and the metric definition. Write down five facts and post them to the team in one message: exact response field names, the metric, the per-item timeout, whether a label file ships, whether audio arrives one consultation per call or batched. Nobody else reads the README; the other three start their stage on the assumption A will answer within thirty minutes.
- **10:30 to 11:00, A.** DTO copied verbatim, twenty-line adapter written by hand, endpoint live behind the pre-built nginx link, **one deliberately degenerate but schema-valid submission through the real validation server**. This proves transport, health check, public link and submission mechanics while there is still time to fix them.
- **10:30 to 13:00, B.** Real audio through the tier-0 config on the event GPU. Measure end-to-end wall clock with a real payload. Run language detection on the real audio: this is the moment the Danish arm activates or dies. Then **the measurement that reallocates the week** (see §8.1): content-term recall on the transcript.
- **10:30 to 13:00, D.** Reimplement the organiser's metric exactly as published. Compute **two floors on the organiser's own examples**: the score of a constant most-common answer, and the score of a copy-the-input answer. If the gap between those and a real system is small, the correct investment for the whole week is house-style matching rather than clinical accuracy, and you need to know that before lunch.
- **13:00 to 19:00, C.** Full transcript into the two-stage extraction against the real DTO. It must beat `prior` (dx_f1 0.0965) and `lexical` (interval 0.4386) or the prompt is wrong, not the model.
- **Hard gate, 22:00.** A valid, scored, non-degenerate submission exists on the validation leaderboard. If it does not, every other workstream stops until it does.

### Day 2, Friday 18 September

The whole day answers one question, and D owns the answer by 16:00: **is the score lost in the transcript or in the extraction?** Report two numbers per field: fraction of gold items present anywhere in the transcript (the ASR ceiling) and fraction that reach the JSON (the extraction loss). Day 3 is allocated entirely by that split.

- **C, morning: build the checklist channel.** One constrained yes/no per candidate label over a cached prefix, threshold tuned on dev. This is the highest-expected-value hour of the event if the split says extraction is the problem.
- **B, morning:** if content-term recall cleared 0.95 on day 1, B moves to extraction and helps C. If it did not, B tunes decode parameters and measures by content-term recall, never by WER alone.
- **C, afternoon:** the verifier pass, one grounded yes/no per unioned candidate with a `rapidfuzz.partial_ratio >= 90` substring assertion on the returned quote. A quote that is not in the transcript is a hallucination and the candidate goes.
- **D, afternoon:** run the ablation that settles the internal argument with a number. Full transcript versus your chunk-retrieve-and-expand design, same model, same prompt, all 57 consultations, field recall with per-consultation win/loss/tie counts. Post it Friday evening so nobody relitigates it on Saturday night.
- **A, all day:** fuzz the live endpoint, then raise the portfolio question explicitly. If Challenge 1 or 2 has no working endpoint by Friday evening, A moves. `survivalsim/` and `serve/api.py` mean Challenge 1 is already half built.

### Day 3, Saturday 19 September

Optimise whichever of ASR loss or extraction loss dominates, and only that one. Three A/B tests, twenty minutes each on the dev set, run by D:

- (a) name the evidence field `evidence_from_transcript` versus `reasoning`. Renaming one JSON key has been measured at +6.9 to -15.8 points of task accuracy under constrained decoding, and the direction is model-dependent, so measure rather than assume.
- (b) emit one condition versus up to three. Gold cardinality is 32/17/7/1, so shotgunning costs precision under any set-F1 grader.
- (c) checklist threshold and verifier drop threshold, swept against the organiser's exact metric. **This is not fine-tuning and it is where most of the measurable gain sits.**

Two rules: **no new components after Saturday lunch**, and A/B against the held-out 15, never against the validation leaderboard.

**18:00: hard freeze.** Tag the commit. Rebuild from pinned requirements in a clean directory. Full dry run against validation with the exact frozen artefact from a **cold process start**, so the lazy-loading bug surfaces here rather than during the single evaluation attempt. Two runs must produce byte-identical output.

### Day 4, Sunday 20 September, ends 16:00 CET

- **09:00:** cold-start validation run of the frozen artefact. Read the public validation leaderboard and make the precision-versus-recall posture call once, on evidence about where the field actually sits.
- **By 13:00:** spend the single evaluation attempt, with three hours of slack. Never in the last hour. There is exactly one attempt per use case against a dataset that differs from the validation set.
- **13:00 to 16:00:** repo hygiene for the Scientific Jury (top five surrender training code and models on a roughly 24-hour clock). Pinned dependencies, a README that reproduces the run from a clean checkout, weights referenced by revision hash, and a grep of the inference path for any http or cloud SDK call. Then every remaining hour into Challenges 1 and 2.

**On the portfolio question.** The 2025 ladder was 25/18/15/12/10/8/6/4/2/1 with rank 11 and below scoring between 1 and 0, summed across use cases. 2026 rules are not published and the aggregation has already changed twice in four year-transitions. Under a sum-with-a-flat-tail ladder, the two things that matter are (i) never have a missing submission, and (ii) concentrate where you can reach the podium. A mid-table third entry is worth almost nothing. Challenge 3 is the cheap podium here because you have a labelled dev set, a measured ASR baseline and a scorer that nobody else will have on day 1.

---

## 6. Decision points, keyed to what the schema turns out to be

Read these in the first thirty minutes and branch.

**D1. Is the required output a transcript scored by word error rate?**
The official blurb ("figure out what was actually said between clinician and patient") is transcription-shaped language, so this is a live possibility rather than a tail risk. If yes: everything after stage [1] switches off with one environment variable, the transcript is the answer, and the whole contest becomes decode-parameter tuning. Immediately ask on Discord which text normalisation the metric uses (jiwer defaults? lowercasing? punctuation? number formatting?), because normalisation routinely moves the number by several points and your error profile makes it bite hard. Then spend the remaining days on ASR only.

**D2. Does a label file ship?**
If yes and it is small (under ~200 entries): build the checklist from *their* labels, constrain hard, and never let a free-text disease name reach the wire. If yes and it is large (ICD-10 is 74,719 codes): do **not** put it in a schema enum, because vLLM throughput collapses to 7.5 req/s at K=1000 and XGrammar compile hits 2.7 s at K=100k. Use the embedding shortlist to get 20 to 50 candidates per consultation and constrain to that per-request enum, after measuring compile time. If no file ships: open free strings plus `canon`, exactly as designed.

**D3. Is the label an integer index or a string?**
The 2025 emergency-healthcare-rag precedent returned `{"statement_is_true": int, "statement_topic": int}` indexing a shipped `topics.json`. If a topics-equivalent ships, mapping to it is a `json.load` plus a dict, twenty minutes. Write the emitter so the label space is a config value rather than a code path, and do not pre-commit to int.

**D4. What is the per-item timeout, and is it per item or per batch?**
Ambolt's published budgets scale with the size of one unit of work: 1 s per simulation tick, 5 s per sentence, 10 s per small image, 30 s for 1000 reviews, 60 s for 2000 reviews. Three of twelve prior cases published none at all. Pick the ASR tier from the answer. Anything above 20 s and tier 0 is fine; 10 to 20 s and you want tier 1; under 10 s and you want tier 2 plus dropping the checklist to the top 20 candidates; under 3 s and the LLM leaves the path entirely.

**D5. Does the request carry one consultation or many?**
If many, batch ASR (batch 8 gives roughly 383x realtime against 219x at batch 1 on the reference numbers). If it is a **stateful polled episode**, which is how race-car 2025 and traffic-simulation 2024 worked, then you accumulate turns across calls, re-run the projection each time, and return the current best answer. The never-delete record makes this survivable; the full-transcript-in-one-prompt assumption does not. Detect this on day 1 from the request DTO, and do not build a batching layer speculatively.

**D6. Is the audio Danish, and is it synthetic?**
Danish: activate the pre-downloaded arm, accept that the alias table, trigger lists and interval grammar are English-only and need about three hours of translation. Synthetic TTS from a written script: WER will be far below 0.139, the deletion bias may vanish, overlap and backchannel handling becomes dead weight, and exact-match scoring on planted facts becomes trivial for the organiser. In that case the real lever becomes reverse-engineering the generator's phrasing from the validation set, and your ASR tuning was aimed at the wrong distribution.

**D7. Does the schema contain an evidence, provenance, turn-index or timestamp field?**
Populate it on day 1. The numbered `L14 [08:31]` transcript already gives it to you for free. On the nearest published benchmark the provenance field was 25% of the leaderboard score and most teams skipped it entirely.

**D8. Does a `reason`, `explanation` or free-text field exist?**
Cap it at roughly gold length. Free-text overlap metrics punish verbosity: adding six filler words to a four-word gold description collapses the score from 1.00 to about 0.56. Extract, do not paraphrase, and do not run a terminology normaliser over an extracted span.

---

## 7. Cut order, agreed in advance

1. The fine-tuned checklist head (a multi-label encoder on 57 examples lands around 0.30 to 0.40 F1).
2. All fine-tuning, including any ASR LoRA. Below ~100 organiser-labelled pairs, use similarity-retrieved exemplars instead.
3. GLiNER-BioMed as a third recall channel.
4. The verifier pass (trades precision for recall, which is the right direction to fail).
5. Speaker diarization and any speaker field.
6. The checklist channel itself, shrunk to top-20 then dropped, if latency bites.
7. Stage [3a], collapsing to a single constrained call or to vLLM's `structural_tag`.
8. Qwen3-14B down to 8B, then 4B. Config change, same grammar.
9. Last resort: serve `urgency`, `modality` and `interval` from the lexical rules alone and leave conditions empty. That already gets interval 0.4386 with zero under-called emergencies, which clears any trivial organiser baseline.

**Never cut:** the ingest adapter, both health routes plus warm-up, the timeout wrapper with its schema-valid default, output self-validation, the deterministic interval normaliser, the emergency floor, and the day-1 validation submission. Together they are under a day of work and they are the difference between a low score and a zero.

---

## 8. What to measure

### 8.1 The one that allocates the week, run before lunch on day 1

```python
from medvoice.metrics import term_recall
# terms = canonical diagnosis tokens + drug names + interval phrases from labels/
found, total = term_recall(reference_transcript, asr_hypothesis, terms)
```

Report content-term recall, not WER. Your `worst_words` says 49.6% of the substitution-plus-deletion error mass sits on 20 function-word and backchannel types, and backchannel filtering moves WER by only 5.3% relative. **If content-term recall clears about 0.95, ASR is not the bottleneck, the entire ASR-recall workstream is dead weight, and B moves to extraction that afternoon.** Nobody in the panel material ran this, and three of the four designs would have spent a day tuning decode parameters against an error mass made of "ok" and "yeah".

### 8.2 Standing dashboard

- **End-to-end wall clock**, p50 and p95, per stage, cold and warm, from the first commit. Roughly 30% of submissions in a comparable served competition failed on time rather than on quality.
- **Schema validity rate**, which must be exactly 1.000, and the repair-loop fire rate, which should be under 2%. A repair loop that fires often is masking a grammar bug you will not find on Sunday. Keep the `jsonschema.validate` assertion even with constrained decoding: XGrammar declares full JSON Schema coverage and then under-enforces in most test-suite categories.
- **Candidate recall before verification**, the Match-style upper bound: fraction of gold conditions proposed by the union of the channels, ignoring precision entirely. Treat anything below 0.85 as the thing to fix and stop tuning precision until it clears.
- **`dx_precision`, `dx_recall` and `dx_f1` reported separately**, never as one figure, with mean predicted cardinality alongside the gold distribution, so you can see whether a prompt change bought recall by padding.
- **`urgency_undercall` as a standalone number** next to `urgency_exact` and `urgency_cost`. Calling an ambulance when the truth was a one-week review wastes an appointment; leaving a stroke at home for a week is the failure that ends a company.
- **Interval accuracy split by null versus value.** 16 of 57 are null.
- **Emergency floor precision and recall** on the 5 gold emergency cases, on its own.
- **Determinism.** Same audio twice, byte-identical output. One assertion before every validation submission.
- **Per-class coverage on every categorical field.** Ambolt has used a product-of-per-class-accuracies metric before (2024 cell-classification), where never predicting one class zeroes the whole score. Assert no field ever returns a constant across the dev set.
- **Grammar compile time** at server start and, if any enum is per-request, per request.

### 8.3 The honesty check

Run the pipeline on consultation A, then score its output against consultation B's labels. If the scorer cannot separate right-patient from wrong-patient by a wide margin, it cannot rank two prompt variants either. Your note work already showed this failure at the ROUGE level (single-reference 0.1782 against a multi-reference max of 0.3211), and it is a structural property of the metric family rather than a quirk.

Also: with n=57 and 92 diagnosis mentions across 57 distinct canonical labels, **the standard error on any single field is roughly six points**. A three-point prompt improvement is not an improvement. Report per-consultation win/loss/tie counts, not just means.

---

## 9. Twenty ideas that did not make the architecture, each priced

You said you would rather have 100 ideas with 10 good ones. Here are the ones that survived scrutiny but lost on schedule, with an honest price and an honest probability.

**Worth an hour, high expected value**

1. **Strip backchannel and filler before the prompt.** ~15 lines. This is the defensible version of your "less junk" instinct, supported by FLenQA's 0.92 to 0.68 drop from irrelevant padding. Measure the token reduction on a real transcript first; if it is under 10%, skip it.
2. **Overlapping ASR windows, 5 s context each side, instead of hard 30 s cuts.** One decode parameter. Measured elsewhere at +30% relative on PERSON entity F1 and numeric character error rate from 0.22-0.42 down to 0.08-0.18. This is your ±5 instinct, at the layer where it actually recovers information.
3. **`initial_prompt` biasing lexicon.** Feed Whisper the 60-entry alias table plus common UK GP drug names. Free, one string, and it targets exactly the content words a biasing lexicon is for. Measure with content-term recall, not WER.
4. **Schema-key wording A/B.** `evidence_from_transcript` versus `reasoning`. Twenty minutes. Measured range is +6.9 to -15.8 points, model-dependent, so it is worth the measurement precisely because you cannot predict the sign.
5. **Two floor references on the organiser's own examples** (constant answer, copy-the-input answer) in the first hour after the metric is visible. Thirty minutes, and it can redirect the entire week toward style matching.
6. **Probe the grader's precision posture with two real variants**, a narrow condition list and a wide one, on separate validation submissions. Validation attempts are unlimited. This is the one grader question your local data cannot answer, and it decides the whole submission.
7. **Logprob-based abstention on the condition field.** You already get the yes-probability from the checklist. One threshold, swept on dev. Under a global micro-F1 grader this is the difference between a good run and a damaged one.

**Worth two hours, real but conditional**

8. **Example retrieval, if the organiser ships gold pairs.** Embed their example inputs, retrieve 1 to 3 nearest by dialogue similarity into the prompt. The MEDIQA-Chat 2023 winner measured about +4 average points from this, with most of the benefit from the first example, so it survives a tiny example set. It is also the cheapest possible purchase of house style.
9. **Example retrieval from your own 57 labels**, testable before the event. Same machinery, no dependency on the organiser. Test it Tuesday.
10. **GLiNER-BioMed as a third recall channel.** Sub-1 GB, millisecond-scale, decorrelated from the LLM, takes entity type names at request time so it absorbs a schema surprise. 59.77 micro-F1 zero-shot across eight biomedical NER sets against 53.81 for the general GLiNER. English-only, so it dies on the Danish branch.
11. **Self-consistency on the condition field.** Three samples at temperature 0.7, union for recall, majority for precision. Attacks omission directly at the cost of 3x on stage [3a] only. Cheap to test, expensive to serve.
12. **Constrained beam search, k=4, for an n-best list of complete structured outputs.** LM Format Enforcer is the only engine with first-class per-beam filtering. On a 50-token output, k=4 beams cost almost nothing against a 2,050-token prefill, and an n-best list is a recall instrument, which is exactly what an omission-dominated task wants. The panel dismissed this as irrelevant at 50 tokens; that inverts the cost model, because beam cost scales with *generated* length.
13. **Patient speech versus clinician speech ablation.** The One in a Million study of real UK GP consultations found problem-category F1 rising from 0.45 to 0.55 when patient speech was included alongside the GP's. If you ever consider filtering by speaker, this says do not.
14. **vLLM `structural_tag` single-pass, reason freely then constrain at a trigger.** Collapses the two-call design into one if the latency budget bites, as a config change rather than a rewrite, while preserving the reason-then-commit ordering that all the evidence says matters.

**Worth two hours, low probability, catastrophic if it hits**

15. **The Danish rehearsal** (already in the pre-event list, item 10). Synthesise five Danish transcripts, measure. Ambolt's `speaches` fork with a Danish `csm-1b` TTS executor, archived seventeen days before the event, is the only direct 2026 artefact about this challenge.
16. **A sequential windowed sweep as the long-audio contingency.** Write the note, not the code. If the organiser hands hour-long or multi-visit audio, the answer is to read *every* window exactly once, one field per call (recall 1.0 by construction, same total tokens as one full-context call, no ranking stage to fail). It is not to restore top-k retrieval. There is no k that is safe: small k loses the chunk, large k reintroduces the whole document plus a near-miss failure mode.

**Interesting, probably cut**

17. **Gemma 4 E4B or 12B audio-native extraction, skipping the text bottleneck entirely.** Genuinely differentiating if it works, since every other team will be running ASR then LLM. Against it: no published clinical evaluation, and the closest proxy has audio-source value accuracy collapsing to about 0.22 against 0.83 on text. Two hours on day 3 if calm, otherwise never.
18. **A fine-tuned multi-label checklist head** (ModernBERT-large or BiomedBERT) on the 57 labels plus any organiser train split. The comparable number is PubMedBERT at F1 0.45 on real UK GP transcripts with 191 training examples. At 57 examples expect 0.30 to 0.40, which is not competitive standalone but is a cheap decorrelated third vote. Only with a verifier downstream.
19. **ICD-10 mapping via embedding retrieval plus LLM rerank** (top-1 accuracy 93.1% versus 71.3% retrieval-only in the published version). Build only if their label file ships codes. Note the cheap hedge that works under both flat and hierarchical grading: emit the most specific label you are confident in, because ancestor expansion gives a specific correct answer credit for all its parents while a vague parent-level answer earns nothing under flat exact match.
20. **Confirming why per-channel decoding fails.** Thirty minutes of diagnostic, zero minutes of building. Your numbers say both deletions and substitutions blow up, not just insertions, which means the cause is not silence hallucination and a VAD-gated merge will not rescue it. Worth the half hour purely so nobody proposes it at 2am on Saturday.

**Explicitly do not do**

- Fine-tune Whisper. Scale cut substitutions 2.3x and deletions only 1.33x from tiny.en to large-v3, so fine-tuning moves the smaller half of your error, on 9 hours of proxy-domain audio, and is strictly negative if the event audio is Danish or synthetic.
- Fine-tune anything below ~100 organiser-labelled pairs.
- Use a medical-domain fine-tune for the extraction stage. Medical adaptation beats its own base model in 12.1% of cases, ties in 49.8%, and loses in 38.2% once prompts are optimised per model.
- Let the LLM do calendar arithmetic.
- Build reflective DTO binding.
- Build a vector index over transcript chunks.

---

## 10. One-paragraph version for the team channel

Keep the ASR front end and mix the channels down (per-channel is 0.3945 WER against 0.1385 mixed, worse on every error type). Delete the chunker, the chunk embeddings, the keyword retrieval and the ±5 expansion: at 15 chunks per transcript, one hit expanded ±5 already returns 73% of the document, so the stage costs an embedding model and buys zero tokens while adding an 8 to 12% omission rate to a task where omissions are already 64% of critical errors. Feed the whole transcript, numbered by line, into one unconstrained pass that lists candidates with quoted lines, run a second pass that asks yes or no about each candidate label explicitly (this is the only thing in the design that structurally prevents forgetting), union them, then project into the organiser's DTO with a grammar-constrained ~250-token call whose evidence field comes first. Do all arithmetic and canonicalisation in Python afterwards. Do not use a closed enum for the diagnosis field: our own labels have 57 distinct conditions across 57 consultations with 43 singletons, so an enum caps recall at about 60% before we start. Move the keyword sets to a lexical sweep that annotates and never discards, because that alone already beats the majority baseline on interval by 15.8 points and catches 5 of 5 emergencies. Build the ingest adapter, the serving shell, the deterministic post-processor and the green nginx connection test before 17 September, because none of them depend on the schema. And before anyone touches an ASR knob on day 1, measure content-term recall: half our substitution-plus-deletion mass sits on twenty function words and backchannel tokens, and if the content words are already coming through at 0.95 the entire ASR workstream is dead weight.