# Is there a better model? No. Keep deepset/deberta-v3-large-squad2.

Four researchers went looking and none found a checkpoint that is better for *this* task on evidence that clears your own noise floor. One candidate (flan-t5-xl) has a margin large enough to be worth a single A/B; everything else is inside ±0.030. The two things that would actually move the score are not model choices.

**The premise the brief was built on is wrong.** Two researchers independently tokenised `finetuning/data/contexts.json`: the transcripts are **240–765 DeBERTa tokens (median 367)**, not 1500–3500. That figure is the *character* count (929/1542/3271). Only 5 of 39 conversations exceed 512 tokens. The whole long-context argument evaporates — see below.

## Shortlist

| # | Checkpoint | Params | Context | QA head exists? | Licence | The one reason |
|---|---|---|---|---|---|---|
| 1 | `deepset/deberta-v3-large-squad2` (incumbent) | 434,014,722 | 512 trained | yes, and it is the only one with HF-**verified** metrics (EM 88.09 / F1 91.16) | cc-by-4.0 | Best verified extractor in the field, and the same backbone VPTSL used to win MedVidQA — the one public task scored by mIoU over time |
| 2 | `deepset/flan-t5-xl-squad2` | 2,783,963,138 | 512 nominal (T5 relative buckets, max distance 128) | yes — `T5ForQuestionAnswering`, a real start/end head, EM 88.79 / F1 91.62 | cc-by-4.0 | Only candidate whose margin plausibly clears your ±0.030: +11.5 F1 on adversarial_qa, +4.8 squadshifts reddit, +3.5 AddOneSent, all vs the incumbent. Out-of-domain is the regime you are in |
| 3 | `sjrhuschlee/deberta-v3-large-squad2` | 434,014,722 (identical) | 512 | yes, EM 87.96 / F1 90.78 | **mit** | Zero-code drop-in; better on 5 of 5 squadshifts/AddOneSent splits, worse in-domain and on adversarial_qa. Costs one eval run |
| 4 | `Praise2112/ModernBERT-large-squad2-v0.1` | 395,833,346 | 8192 native | yes, EM 86.27 / F1 89.30 — **self-reported, 42 lifetime downloads** | apache-2.0 | The only drop-in long-context SQuAD2 span checkpoint that exists. Listed so it is explicitly ruled out: `HasAns_F1` 87.07 vs the incumbent's 90.67 — a 3.6 F1 deficit on the exact half of the metric you use |

Ruled out with reasons: **deberta-v3-xlarge/xxlarge do not exist** (v3 shipped only to large; v2-xlarge has no SQuAD2 checkpoint, only CUAD). `ahotrod/electra_large_discriminator_squad2_512` is the most-downloaded QA model on the Hub (768k) and is a 2020 trap — 1.2 F1 below, no licence declared. Both Longformer large checkpoints declare **no licence at all** — that is no grant of rights, exclude before any engineering argument. NeoBERT, Ettin, EuroBERT, GTE-en-MLM, LFM2.5: no QA prior, which is the one thing 195 spans cannot supply.

## Verified vs unverified

**Verified** (live HF API, config.json, model cards, code run locally): all parameter counts, licences, context settings, architectures; every SQuAD2/squadshifts/adversarial_qa number quoted; the token/window statistics on your own data; `ModernBertForQuestionAnswering` native in transformers 5.2.0; DeBERTa-v3 having no absolute position embeddings.

**Unverified — do not launder these:**
- Nobody ran *any* of these checkpoints on your data. Every ranking is inference from published benchmarks. Your own rule (results are size-specific) cuts against positive results too.
- The flan-t5 case rests on squadshifts reddit/amazon being a proxy for spoken medical transcripts. That is an argument, not a measurement. **What would confirm it:** one fine-tune + eval on your 9 folds, ~$5 of A100.
- Praise2112's numbers are a single unreplicated hobbyist run with no OOD eval.
- sjrhuschlee's and deepset's flan-t5 metrics are self-reported; only the incumbent's carry HF verifyTokens, so cross-repo comparisons at 0.5 F1 are softer than the decimals suggest.
- All wall-time and latency figures are FLOP arithmetic — there was no CUDA torch in any session.

**One direct conflict between researchers, and it matters.** Three of them ran DeBERTa-v3 past 512 tokens and reported "it works, delete the sliding window." All three used random weights or a 2-layer stub — they proved the *architecture* accepts the length. The fourth ran **real trained weights** (`deberta-v3-base-squad2`, zero-shot) on your own rows and measured the output **collapsing**: best-span logit +11.05 at max_len 384 → **−3.35 at 768**, with the predicted span moving to an unrelated clause. The relative-position buckets saturate past the pretraining range. Trust the run with real weights. **1024 is not available to you. 512 is.**

## The long-context question, honestly

Native long context buys you **neither accuracy nor, now, simplicity**.

- Not accuracy: 0 of 195 gold spans fall outside the windows, and that is not luck — consecutive windows overlap by exactly stride 128, and the longest gold span is 40 tokens (median 10, p90 20). Straddling is **impossible by construction**.
- Measured cost of the merge logic: shipped merge 0.2883 vs oracle-window 0.3214 character IoU on the 97 gold rows that split, delta +0.0331 ± 0.0306, and the merge changes the prediction on only **9 of 97 rows**. Scaled to all 195 golds that is ~+0.016 — invisible against your ±0.049 instrument. Caveat: measured zero-shot on a *base* model at 0.29 IoU, a much weaker regime than your 0.76+ target, so treat it as directional.
- Not simplicity either, because the windowing is an artefact of the branch's shipped `--max-len 384`, not of DeBERTa's cap. At 384/128, 179/390 rows split (up to 3 windows). **At 512/128, 61/390 split, max 2 windows.** One flag, entirely inside the trained range, zero risk, and it removes most of the merge surface. Going to 8192 to solve a problem whose longest row is 792 tokens is not a trade worth an unaudited checkpoint.

## Is span extraction the right formulation?

Yes, and there is direct precedent. **MedVidQA** is scored by mIoU over time — literally your metric. **VPTSL** (arXiv 2203.06667, TPAMI 2024) won it by localising a *text* span over timestamped subtitles with **DeBERTa-v3-large**, then mapping to time — your pipeline shape exactly — beating the best video-side model by +28.36 mIoU (57.81 vs 29.45), trained on 2,710 questions. Same formulation, same backbone, same order of supervision.

The "SQuAD answers are short" objection conflates the head with the training data. **`allenai/quac`**: mean answer **14.6 tokens** (SQuAD 3.2, CoQA 2.7), 83,568 train questions, MIT, contiguous character spans, a `yesnos` field and a `CANNOTANSWER` class — i.e. "boolean question plus the stretch of text that settles it." That is the better-matched pre-training corpus, and there is **no public QuAC extractive checkpoint** (HF `dataset:quac` returns question-*generation* models), so it is a ~1.5–3 h A100 job, $2.50–$5. `stanfordnlp/coqa` rationale spans are the runner-up on task shape, worse on length.

Do **not** pre-train on QASPER / FEVER / HotpotQA supporting-facts / MS MARCO / LongCite: all sentence-or-paragraph granular by construction. Your clause-and unit oracle is 0.919 mean tIoU against a sentence-unit oracle of 0.861 — a sentence-granular evidence selector is capped *below* the machinery you already ship.

Three things no checkpoint swap fixes: none of these models was ever *selected* for interval placement (all EM/token-F1); all carry the same short-answer prior; and the `chars_to_time` cliff (+3 characters costs 0.052 tIoU) is a property of the char→time snap, not the encoder — SentencePiece and BPE have identical offset semantics, and only 3 of 195 gold spans are exactly representable as token boundaries under either.

## What I would do with one day

Ordered by expected value per hour. Finding 59 still governs: remaining headroom ≤0.024 against ±0.030 resolution, so be honest that most of this is bug-removal and next-year leads, not Sunday points.

1. **Change nothing about the checkpoint.** His choice is correct and I would defend it in review.
2. **`--max-len 512`** (30 seconds, zero risk). 179/390 → 61/390 multi-window, max 2 windows. Do **not** go to 768 or 1024 — that is the measured collapse.
3. **Expected-IoU / MBR decode instead of argmax** (~30 min, no GPU). The head gives p_start and p_end and you take the *mode* of the span posterior; the metric asks for the argmax of expected IoU under that posterior. Same idea on the 27B: finding 50 scored 8 samples by majority/modal/longest/shortest but never by the **IoU-centroid**, and its best-of-samples oracle was 0.8556 against greedy 0.8158. If proposer 5's samples are still on disk this is laptop-only re-analysis. Unverified derivation — no published precedent for text spans; it depends on the edge posterior being calibrated, which 195 spans makes doubtful.
4. **Fix the hyperparameters before reading any result.** At batch 2 × accum 8, single-window encoding gives ~22 optimizer steps/epoch, **~110 steps total at lr 5e-6**. That is the textbook underfitting regime (Mosbach et al., ICLR 2021); a flat result would mean nothing. Raise epochs to 15–20. Also: `--optim adafactor` is load-bearing on 8 GB (AdamW state alone is 7.28 GiB and will OOM), and `padding='max_length'` wastes ~60% of every pass — pad to longest-in-batch for a free 2.4x.
5. **One A/B, if any: `deepset/flan-t5-xl-squad2`.** It is the only margin that might be visible. 5.6 GB bf16 so it will not fine-tune on the 5070; it is a rented-GPU job. Note you do not need an A100 for a 435M encoder — A5000 24 GB is $0.27/h, A40 48 GB $0.49/h.
6. **Do not** rent for a ModernBERT swap, do not build a BIO tagger or a reranker (finding 50 already lost that one), do not add LongCite or a second decoder to a 60 s budget, do not regress offsets.

Two integration costs nobody priced: vLLM has **no span-extraction task** (pooling tasks are embed/classify/score/token_classify/token_embed/reward) and Ollama serves no encoders, so this must be a hand-written uvicorn process — put it in the pod's existing `/opt/venv-vllm`, which already has a CUDA-13 torch, so nothing re-resolves venv-api's pins. And RUNPOD.md records GPU_UTIL 0.85 with 77 of 80 GB in use; a bf16 encoder wants ~2 GB with its context, so a fourth GPU process means lowering GPU_UTIL on a frozen, pre-flight-passing config. That is the real cost of shipping this, not the model choice.