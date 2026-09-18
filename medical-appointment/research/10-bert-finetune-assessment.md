# Nikolaj's extractive-QA fine-tune (`origin/medicalBertFinetune`): assessment

## Bottom line

**Not a submission candidate for Sunday. Worth about 2.5 hours and $2 on Saturday morning to settle, if you want it settled, and only after two changes that cost minutes.**

Nothing on the branch has been trained. The 16 committed files are scripts, data and documentation; `runs/`, `checkpoints/` and `logs/` are gitignored and nothing under them was pushed. I checked the other places a number could hide (Nikolaj's other branch, the stash, the working tree) and found none. So there is no measurement anywhere saying this fine-tune reaches any temporal intersection over union (tIoU, the overlap metric that carries 0.6 of the score) at all. Your hypothesis cannot be answered from the branch. It can only be answered by running it.

What can be settled without a GPU is whether running it is worth the hour, and whether the hypothesis is the right shape. It is not quite the right shape, and that is good news for Nikolaj: **the two halves of the score do not trade against each other.** You never have to accept a worse binary to buy a better interval.

## What is good about it, specifically

- **The data conversion reproduces byte-for-byte.** Re-running `build_qa_dataset.py` unmodified against `data/question_train.csv` and the transcripts produced `qa_train.jsonl` and `contexts.json` identical to the committed blobs, SHA-256 equal. The stored `answer_time_*` round-trip back through the inference-time `chars_to_time` to within 0.017 s, which is two-decimal rounding. `DATASET.md`'s own ceiling table (0.907 word-exact, 0.824 unit-snapped) reproduces to the third decimal.
- **The sliding window is correct, and that is the part easiest to get silently wrong.** He hand-rolled `windows()` (`finetune_qa.py:106-135`) because transformers' `return_overflowing_tokens` silently drops text past two windows, documented why in the docstring, and wrote a regression test for it. Checked directly: 0 of 195 gold spans fall outside every window, `--max-answer-tokens 80` never binds, and only 2 characters across all 386 rows are unreachable by any window. Half the corpus needs multi-window handling (189 of 390 question rows), so this was load-bearing.
- **The fold split is genuinely by conversation.** Verified in code and by re-running `make_folds`: shuffle transcript ids, disjoint test blocks, filter records by membership, an overlap assertion in `main()`, and a unit test on the real 386-row file. 39 distinct context strings for 39 conversations, so no shared-context path either. No row-level leakage exists.
- **Zero contamination.** Nothing in the pipeline can reach `request_dump/` or `bench/mine/`. Verified four ways: the builder reads only `question_train.csv` and `transcripts/`; the transcripts directory holds exactly the 39 training ids and none of the 19 validation ids; all three committed data files cover 39 conversations with zero validation ids; a branch-wide grep for the validation ids and for `request_dump`, `bench/mine`, `portal`, `submit` returns nothing.
- **He gets the denominator right.** Both scorers take the tIoU denominator from the gold labels, not from what the model answered, so a gold-yes question answered no still counts in the mean. That is the trap most re-implementations fall into.
- **He documents what he has not closed.** `README.md:40-41` states outright that the per-epoch held-out numbers are validation results and not an independent test after epoch selection. `DATASET.md` tells the reader to split by `transcript_id`, flags the four rows where Whisper dropped the annotated speech instead of training on wrong spans, and says that switching the speech recogniser requires rebuilding with matching offsets. `committee_eval.shared_questions` refuses to compare members whose held-out ids, folds or golds differ. `--dump-preds` fires only on the final epoch, so the committee cannot epoch-shop on top of its rule search.

This is not the work of someone who would fake a number. It is the work of someone who has not yet produced one.

## Your hypothesis, answered

**The premise of the trade is wrong, in the branch's favour.** `local_evaluator.py:130-145` returns early only on `label != YES`; after that it computes `temporal_iou(gold, predicted)` and appends it whatever boolean was returned. `model.py:129` has `SPAN_ON_NO` defaulting to 1. And this is settled against the live portal: entry 38 records that run F's dumped answers score 0.67591704 under spans-on-every-question against the portal's reported 0.6759170394, and 0.65902734 under nulls-on-no. Entry 15, which said the opposite, is explicitly marked wrong in the log. One of the five reviewers leaned on entry 15 and reached the wrong conclusion; ignore that line of their report.

So the score is additively separable. You can keep the 27B's binaries and take the spans from wherever they are better. **The break-even for that hybrid is identically the incumbent's current mean tIoU, 0.71189.** That is algebra, not a measured coincidence: hold accuracy fixed and the accuracy term cancels. One reviewer reported 0.7122 and called the 0.0002 gap "parity"; that 0.0002 is entirely the rounding of 0.9974 to 0.997, and the correct statement is that break-even sits exactly on the incumbent, for any span source whatsoever.

That makes the question of whether the fine-tune is worse at the binary **irrelevant to the configuration worth testing**, and the answer to "could it be better at the interval" turns entirely on ceilings.

### Ceilings (all recomputed independently, two scripts agreeing)

| quantity | mean tIoU |
|---|---:|
| Incumbent achieved, 39 training conversations | 0.7119 |
| Incumbent's clause-unit oracle (entry 47) | 0.919 |
| Incumbent's turbo word oracle | 0.9451 |
| Branch target learned perfectly, as committed (large-v3, all 195 golds) | **0.8950** |
| same, over the 191 rows the branch keeps | 0.9072 |
| Branch target rebuilt on turbo, all 195 golds | **0.9335** |
| Best contiguous word range on large-v3 with the branch's offsets | 0.9292 |

**As committed, the ceiling argument does not support your hypothesis at all.** 0.8950 sits 0.024 *below* the incumbent's own clause-unit oracle of 0.919. The branch's representation currently has less room than the representation it would replace. Rebuilt on turbo it reaches 0.9335 and clears it, and the rebuild also fixes the four rows where large-v3 dropped 8.9 seconds of speech in `sample_81` (turbo transcribes it; `alignment_ok=false` goes from 4 rows to 0).

Two figures in our own record needed correcting in the course of this. Entry 42's large-v3 oracle of 0.935 is 0.006 high; measured directly on large-v3's own words under large-v3's own rule it is 0.9292, because entry 42 text-aligned large-v3 words onto turbo's word list. The turbo figure 0.946 is confirmed exactly at 0.9451.

### Break-even

| configuration | accuracy | mean tIoU needed to tie 0.8261 |
|---|---:|---:|
| Hybrid: 27B binaries, fine-tuned spans | 0.9974 | 0.7119 (parity) |
| same, to clear the instrument's ±0.030 at 95% | | **0.762** |
| Fine-tune spans + laptop qwen3:4b binaries | 0.990 | 0.717 |
| Fine-tune alone | 0.98 | 0.723 |
| Fine-tune alone | 0.95 | 0.743 |
| Fine-tune alone | 0.90 | 0.777 |

As a fraction of its own ceiling: the incumbent reaches 75.3% of the turbo word oracle. The fine-tune rebuilt on turbo must reach 76.3% of its ceiling to tie the span half and **81.6% to produce a gain we could see**.

### Is 0.762 reachable? The ladder says it is the edge of plausible

Measured on large-v3 with the branch's units and offsets:

| step | mean tIoU |
|---|---:|
| lexical best-overlap unit | 0.4608 |
| perfect sentence pick, no trimming | 0.7066 |
| **incumbent achieved** | **0.7119** |
| perfect word range inside the correctly chosen sentence | 0.7893 |
| perfect unit plus perfect neighbour decisions | 0.8064 |
| perfect word range anywhere | 0.9292 |

Perfect sentence selection alone lands *below* what the incumbent already gets. Every cent of gain has to come from sub-sentence trimming. And 0.7893 (perfect trim given perfect sentence choice) reproduces entry 44's 0.787 to 0.795 by a different route, so the detectability bar of 0.762 sits at 96% of what an oracle trimmer would achieve with oracle sentence selection.

**The one genuine mechanism argument in the branch's favour**, and it is a real one: character-level boundaries are the only thing measured that can reach the inside-the-unit headroom of 0.118 (entry 45). The learned re-ranker failed (entry 50, 0.694 to 0.688-0.694) and the Sonnet probe matched us rather than beating us (0.6845 against 0.686), but both of those re-select over whole units and structurally cannot trim inside one. Your own rule applies here in Nikolaj's favour: entry 50's negative was measured with ridge and gradient-boosted trees over 55 hand features on 195 rows, not with an encoder carrying a SQuAD2 prior that reads the raw text, so it does not settle this.

**The counterweight**, also measured: the annotators' neighbour-inclusion decision is predictable from the text (leave-one-conversation-out logistic regression, AUC 0.907 for the previous utterance, 0.789 for the next), but converting that into tIoU nearly fails. Including a neighbour correctly is worth +0.36, including it wrongly costs −0.27, so the break-even posterior is 0.43 and the classifier clears it on 12% of candidates. Of the 0.095 tIoU available from perfect neighbour calls, prediction captures 0.009, or 0.015 at a swept and optimistic threshold. Speaker change, one of the three features you named, is null in both directions (z −0.58 and −1.52).

### On the binary half

Yes, it will almost certainly be worse, and for a structural reason rather than a tuning one. There is no yes/no head: it is `AutoModelForQuestionAnswering`, start and end logits only, with the binary derived purely from `min_null - best_span_score > null_threshold` and the threshold pinned at 0.0. 142 of the 195 "no" questions are hard negatives, where a textually perfect span exists and only a value or polarity comparison denies it ("Is the pulse recorded as 92?", "Does the fasting blood sugar lie at 5.0 mmol/L?"). Span scoring has no mechanism for that; our 27B does, and gets 142/142, as does qwen3:4b.

Scoped per your rule, the only measurement is size-specific and zero-shot: running `deepset/roberta-base-squad2` through the branch's own encode and postprocess gave accuracy 0.741 overall, off-topic 52/53 but hard negatives 95/142. Sweeping the null threshold on the test data itself bought only +0.013 accuracy, which suggests the CLS margin does not separate hard negatives at any threshold. That is base RoBERTa with no fine-tuning; it does not bound what a fine-tuned DeBERTa-v3-large does after seeing ~128 in-domain hard negatives per fold, and no run exists to say.

None of this matters in the hybrid, where we keep the 27B's booleans.

## Do these two things before spending a GPU hour

**1. Rebuild on turbo.** `build_qa_dataset.py:46` is `ASR_TAG = 'large-v3'` and `:38-39` are the large-v3 offsets. Rebuilding with `large-v3-turbo` and (−0.20, −0.02) takes the perfect-target ceiling from 0.8950 to 0.9335 over all 195 golds, worst row from 0.250 to 0.540, and unaligned rows from 4 to 0. All 39 turbo transcripts are already present. I tried to break this by arguing the offsets were doing the work: grid-searching both edges over −0.60 to +0.60 at 0.02, in-sample and leave-one-conversation-out, large-v3's best possible offsets recover 0.0007 of the 0.0384. The gap is the timestamps. Transcript text quality is indistinguishable between the two (token error rate 0.0579 against 0.0578 on the 17 hand-corrected references).

**This is not a three-line change.** `finetune_qa.py:36-37` carries a second copy of the same two constants, used by `to_time()` at `:94-95`. Change only the builder and you get turbo contexts mapped through large-v3's constants, measured at 0.8904, which is worse than the committed 0.8950. The real change is 5 lines across 2 files, plus regenerating the three data files and updating `DATASET.md`'s ceiling table. `DATASET.md`'s "Known limits" already says both files must change; Nikolaj documented it.

**2. Widen the folds.** `--folds 2 --test-frac 0.1` holds out 4 conversations per fold, 8 of 39 in total, 80 questions, 45 gold positives, and never evaluates the other 31.

| held out | 95% interval on score | on mean tIoU |
|---|---:|---:|
| 39 (the incumbent's bar) | ±0.030 | ±0.049 |
| 12 (`run_committee` default) | ±0.053 | |
| 8 (`finetune_qa` default) | ±0.065 | ±0.107 |
| 4 (one fold) | ±0.092 | ±0.153 |
| 36 (`--folds 9`) | ±0.031 | |

At the shipped default a single fold cannot distinguish a span quality of 0.65 from 0.80, which is the entire range in dispute. `make_folds` already permits `--folds 9` at test-frac 0.1 (nine disjoint blocks of 4, covering 36 of 39). The fix is free and needs no new code.

## What would embarrass us if it shipped unexamined

1. **The character-to-time cliff.** `chars_to_time` filters on `w['char_end'] > cs` and anchors on `ws[0]['end']`, so two characters of leftward slop at the span start picks up the *previous* word and anchors on its end. Mean tIoU collapses from 0.9073 to 0.7318, on 188 of 191 rows, worth −0.105 of score. The error is asymmetric: −1 character is harmless, +3 characters costs 0.052. There is no snap-to-word-boundary guard. The incumbent cannot hit this because `span_from_ids` only ever emits whole units. This is the single scariest thing on the branch for serving and it is a few lines to fix.
2. **Comparing any held-out number to 0.8261.** Scoring the incumbent's served run on exactly the conversations these folds hold out gives 0.8203 as the branch's own script would report it (unweighted mean of folds, and the two folds carry 24 and 21 positives), 0.8238 pooled, and 0.8338 at `--folds 3`. The three 4-conversation folds score 0.8677, 0.7730 and 0.8603 for one unchanged system, a 0.095 spread.
3. **A headline picked by best epoch, then best committee member, then best fusion rule, all on the same 80 to 120 questions.** By entry 59's own formula, rescaled, that carries +0.017 to +0.021 of expected optimism, roughly 1.7 times what we charged ourselves over ten variants, entirely because the set is a third the size. `committee_eval` then prints categorical sentences ("VERDICT: not worth it") at thresholds of 0.005 and 0.02 that both sit inside a ±0.018 paired noise band.
4. **`finetune_qa.py:59` drops the 4 unaligned rows from evaluation as well as training**, so the self-reported mean tIoU rests on 191 golds while the portal scores 195, about 0.012 of self-flattery. Harmless under the shipped seed (neither `sample_57` nor `sample_81` reaches a held-out block until `--folds 5` and `--folds 9`), but seed-fragile.
5. **`DATASET.md` calls 0.907 and 0.824 "the ceilings, what a model scores if it is always exactly right".** They are the score of reproducing his own training target exactly. The representational ceiling is 0.9292. That label will get quoted later as if it bounded the approach. `DATASET.md:150` also says large-v3 is "the served pipeline's default", which is wrong; we serve turbo. And the "0.450 for off-the-shelf roberta-base-squad2" is our entry 23, measured on turbo with unit snapping, not on his data; it is the document's only external anchor and it flatters the gap it claims to leave open.
6. **Putting torch into `venv-api`.** `pod_endpoint.sh:27` installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` unpinned and points `LD_LIBRARY_PATH` at them so ctranslate2 can find CUDA. Installing torch into the same virtual environment re-resolves exactly those two packages, and if the resulting cudnn is not the one ctranslate2 was built against, faster-whisper loses CUDA at the next supervisor restart and the endpoint serves with no speech recognition at all. Run the QA model as a third local process behind an HTTP call, the pattern vLLM and Ollama already prove on that pod.
7. **Serving it as committed needs large-v3 alongside turbo**, at RTF 0.111 against 0.043, roughly +13.6 s on the mean conversation, which is 149% of the current end-to-end round trip. The turbo rebuild removes this, because `model.py` already holds the turbo word list with timings and the joined context can be built in-process.

## Two smaller things, for the record

`gold_coverage` is a weak gate but not a useless one, contrary to how one reviewer put it. Its entire informational content is the binary "the gold interval contains no transcribed word": coverage exactly 0.0 flags precisely the two rows whose reconstruction tIoU is 0, with perfect precision and recall. Above zero it is flat (r = −0.0003 over the 193 rows with at least one overlapping word). The real defects are that `MIN_COVERAGE = 0.5` wrongly excludes `sample_57_yes_q06`, whose conversion is perfect at tIoU 1.000 and whose coverage is 0.492 only because the gold interval contains two genuine pauses, and that it cannot see `sample_63_yes_q02` and `sample_64_yes_q02` at all (coverage 1.000, tIoU 0.250 and 0.267: sub-0.3 s gold intervals at audio onset annotated over the word "Good"). Those last two are the case's own annotation defects and cap both pipelines equally. The one-line improvement is to gate on the reconstruction tIoU the builder already computes at `:189-190`, which drops all five bad rows, keeps 190, and raises the reported ceiling to 0.9146. Footprint of the whole issue: 4 rows of 390.

Serving cost is not an obstacle. One conversation is 15.6 windows of 384 padded tokens (max 30), about 5.0 TFLOP for one DeBERTa-v3-large forward, roughly 0.1 to 0.2 s on an idle A100 and an estimated 0.5 to 2 s on the laptop card. I could not time a real forward pass (no GPU in this session, no weights cached), so treat the laptop figure as an estimate rather than a measurement, and note it is 5% to 22% of our end-to-end depending on hardware, not "negligible" on the laptop path. If the three-member committee of `run_committee.py` were served rather than the best single member, every figure triples.

## Recommendation

**Submit the frozen incumbent.** It is validated at 0.8084, the pre-flight passes, and we have one evaluation attempt unspent.

**If you want the question settled for the record, here is the experiment, about 2.5 hours and $2, decision by Saturday midday.** It is span-only: ignore the binaries entirely, because in the hybrid we keep the 27B's.

1. *30 min, laptop, no GPU.* Rebuild the dataset on turbo: `ASR_TAG` plus the four offset constants across `build_qa_dataset.py` and `finetune_qa.py`, regenerate `qa_train.jsonl`, `qa_train_squad.json` and `contexts.json`.
2. *45 min.* Two harness fixes, before any number is produced, because a number produced before them has to be discarded rather than adjusted: snap the predicted character start to a word boundary (the cliff above), and always emit the argmax span regardless of the boolean, which `postprocess` already has in hand at `finetune_qa.py:221`. The second matters less than one reviewer claimed, since `char_start`/`char_end` are dumped unconditionally at `:243-244` and the always-credit tIoU is recomputable offline from `preds.jsonl` plus `contexts.json` with no GPU, but `committee_eval.load()` requires `time_start`, so fix it at the source.
3. *~1 hour, one A100 or a cheaper L40S, about $1.60.* `--folds 9 --test-frac 0.1 --epochs 5 --dump-preds`. Fix the epoch count in advance and read the final epoch, which `--dump-preds` already pins at `:393`. Do not run `committee_eval`; three members on 120 questions buys nothing but optimism.
4. *20 min, laptop, no GPU.* Score the hybrid offline by pairing the held-out spans with the 27B's preserved booleans from `bench/results/served/pod3/answers.jsonl`. `bench/llm/replay.py` already does this shape of rescoring.

**Pre-commit the bars now, before seeing the number:**

- below **0.762** mean tIoU on training: indistinguishable from what we have on a 39-conversation set. Record it in the findings log and stop.
- 0.762 to 0.79: real but not worth swapping a frozen, validated pipeline the night before a deadline, because the incumbent dropped 0.018 from training to validation and a new component measured with epoch selection on a smaller set would eat that margin. Record it as a lead for next year.
- above **0.79**: worth the integration conversation, and worth noting that 0.79 is the perfect-trim-given-perfect-sentence-selection figure, so clearing it would be a genuinely surprising result from a 435M encoder trained on 195 positives across 35 conversations per fold.

Integration and a validation run cost another 3 to 4 hours on Saturday and re-open a pre-flight that currently passes. That is the honest reason this is not a submission candidate: not that the idea is wrong, but that the evidence does not exist yet and the calendar does not allow producing it *and* validating a changed pipeline.

One last thing worth telling Nikolaj plainly: the reason this is a close call rather than an easy no is that his approach is the only one we have tried that can address sub-unit boundaries at all, and our two strongest negative results on span quality (the re-ranker and the Sonnet probe) are both scoped to systems that re-select over whole units. He built the right thing. He built it on the wrong transcripts, and he has not run it.

*Provenance: every number above was recomputed in the scratchpad from the repository's own inputs, read-only. No repository file was modified, no branch was checked out. Two independently written scripts agree on the oracle figures. One reviewer reported that their scratchpad `oracle.py` was rewritten by something outside their session between runs and that the replacement would have raised a NameError; they did not run it and re-derived the same quantities with a fresh script, which is why the oracle figures are quoted from the duplicated run rather than the original.*