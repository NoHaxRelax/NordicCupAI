# Extractive QA training set — medical-appointment

390 questions over 39 consultations, shaped for span extraction. Built by
`scripts/build_qa_dataset.py` from the Nordic AI Cup `medical-appointment` case:
its `data/question_train.csv` and its Whisper `large-v3` transcripts.

Rebuild with:

```cmd
py scripts/build_qa_dataset.py
```

## Why this file exists

The case annotates evidence in **seconds of audio**. A span extractor works in
**characters of text**. This dataset is that conversion, done once: each
conversation's word timings are joined into a single context string, and each
gold time interval becomes the character range of the words it covers.

## Files

| File | What it is |
| --- | --- |
| `qa_train.jsonl` | 390 records, one per question. The main file. |
| `qa_train_squad.json` | The same, in the nested SQuAD-v2 layout `run_qa.py` expects. |
| `contexts.json` | Per-conversation char↔time alignment: every word and unit with its offsets and times. |

## Record shape

```json
{
  "id": "sample_4_yes_q03",
  "transcript_id": "sample_4",
  "audio_file": "conversation_sample_4.mp3",
  "duration": 106.48,
  "question": "Did the patient attend for an annual asthma follow-up?",
  "context": "Morning, Dr Fabricius. Christian, come in. ...",
  "answer_yesno": "yes",
  "label": 1,
  "question_type": "positive",
  "is_impossible": false,
  "answers": {"text": ["So, this is your annual follow-up. It is, for the asthma. Exactly"],
              "answer_start": [311]},
  "answer_end": 376,
  "evidence_start": 21.62, "evidence_end": 26.24,
  "answer_time_start": 21.72, "answer_time_end": 27.24,
  "answers_unit": {"text": ["So, this is your annual follow-up. It is, for the asthma. Exactly that."],
                   "answer_start": [311]},
  "answer_unit_end": 382,
  "unit_ids": [8, 9, 10],
  "unit_time_start": 21.72, "unit_time_end": 27.9,
  "gold_coverage": 0.848,
  "alignment_ok": true
}
```

`evidence_*` are the case's own gold seconds. `answer_time_*` are what this
record's character span maps back to. The `answers_unit` variant is the same
span widened to whole sentence-ish units.

Negatives (`hard_negative`, `off_topic`) carry `is_impossible: true`, empty
`answers`, and none of the span fields — matching SQuAD v2 unanswerables.

## Composition

| | count |
| --- | --- |
| positive (answer yes, has a span) | 195 |
| hard_negative (answer no) | 142 |
| off_topic (answer no) | 53 |
| **total** | **390** |

Contexts run 178–630 words (median 292), 898–3312 characters. Answers run 5–167
characters (median 44). The longest contexts exceed 512 tokens, so a BERT-family
model needs a sliding window (`--stride`), or use a long-context encoder.

## Two targets, and which to train on

Each positive carries two versions of the same evidence:

- **`answers`** — word-exact. The words the gold interval actually covers.
  Truncates mid-sentence, as in the example above where it stops at "Exactly".
- **`answers_unit`** — snapped out to whole units, the same sentence-ish units
  `medical-appointment/model.py` builds. Always complete sentences.

Mapped back to seconds and scored against the gold interval, these are the
ceilings — what a model scores if it is *always exactly right*:

| target | mean tIoU | min | below 0.5 |
| --- | --- | --- | --- |
| word-exact | **0.907** | 0.250 | 2 / 191 |
| unit-snapped | 0.824 | 0.184 | 21 / 191 |

Word-exact is the higher ceiling, so it is the default target. Unit-snapped is
the easier thing to learn and the more natural output; the table is the price of
that choice. For reference, off-the-shelf `deepset/roberta-base-squad2` scored
0.450 on this data, so there is a wide gap to close.

## Mapping a prediction back to seconds

This is the part that is easy to get wrong, and it is worth 0.6 of the case
score. A predicted character span has to become `evidence_start` /
`evidence_end`. Use `contexts.json`:

1. Find the words whose `[char_start, char_end)` overlap the predicted span.
2. `start = first_word.end - 0.14`
3. `end = last_word.end + 0.12`

Those offsets are not arbitrary. The annotators' interval begins near the **end**
of its first word, not its onset — anchoring at `first_word.start` instead reads
about 0.3 s early on every span and costs ~0.09 mean tIoU. They are the
leave-one-conversation-out values fitted for `large-v3` in
`medical-appointment/model.py`.

## Four rows you should exclude

`sample_57_yes_q06`, `sample_81_yes_q02`, `sample_81_yes_q03`,
`sample_81_yes_q04` have `alignment_ok: false`.

Whisper dropped the speech those questions are annotated on — `sample_81` has an
8.9-second hole between 59.98 s and 68.88 s, and the gold evidence for two of its
questions (63.88–68.32) sits entirely inside it. The span is not in the text at
any offset, so their `answers` point at the nearest surviving words, which is
wrong. They are kept in the file for completeness and excluded from every number
above.

**Filter on `alignment_ok` before training**, or you teach the model four
confidently wrong spans:

```python
rows = [r for r in rows if r['is_impossible'] or r['alignment_ok']]
```

Re-transcribing those two conversations, or correcting them against
`bench/ref/`, would recover the rows.

## Splitting

The file is all 390 examples, unsplit. **Split by `transcript_id`, not by row.**
Ten questions share each context, so a random row split puts the same
conversation on both sides and the score you measure will not survive contact
with the evaluation set.

## Known limits

- Contexts are ASR output, not ground truth. Only 7 of the 39 conversations have
  human-corrected references (`bench/ref/`, the ones marked `# checked`), and
  those carry segment markers rather than word timings, so they are not
  drop-in replacements.
- `large-v3` is the transcript set here and the served pipeline's default.
  `large-v3-turbo` has its own fitted offsets (−0.20 / −0.02). If you switch,
  rebuild with a matching `ASR_TAG` and update the conversion offsets in both
  the dataset builder and training script.
- Answering yes/no and locating the span are one task in the case but two
  columns here: `label` for the first, `answers` for the second. Nothing in this
  file forces a model to do both.
