# Hand-correcting eight reference transcripts

One person, 45 minutes. The result is eight files in this folder whose first
line reads `# checked`.

`bench/asr/compare.py` reports the word error rate (WER, the share of reference
words a transcript gets wrong, counting substitutions, deletions and insertions)
for every automatic speech recognition (ASR) tag. Without hand-corrected files
it can only score one model's text against another model's, which measures how
much two models disagree, not which one is right. Eight corrected files give a
small gold subset; the table then shows `WER ref (n=8)` next to the silver
number.

## What is already here

`bench/ref/conversation_sample_*.txt` exist for all 39 training conversations,
pre-filled with Whisper large-v3 text, one segment per line, each line starting
with a `[m:ss.s]` marker for finding the spot in the MP3 (`INDEX.md` lists them).
The scorer strips the marker and only counts a file whose first line is
`# checked`. An unchecked file is Whisper text and would score WER 0 against
Whisper, so the scorer skips it. It also drops, with a warning on stderr, a
file that carries `# checked` but whose words still equal one model's
transcript after normalisation: such a file was marked, not corrected, and
counting it would report that model at 0 percent and score every other model
against Whisper instead of against a person. Add the marker only after
listening; a file with punctuation edits alone still counts as unchanged.

On 2026-09-17 `conversation_sample_4.txt` was in that state: marked
`# checked`, word-identical to `large-v3`. If you did not listen to it,
remove the marker line; if you did and Whisper was right throughout, say so
in a `# ` comment line under the marker so the next person knows, and
expect the scorer to keep dropping it until at least one word differs.

## Which eight

```
python bench/asr/compare.py --suggest-ref 8
```

This ranks the 39 conversations by how many of their ten questions contain a
digit, breaks ties by shortest audio, and takes eight within 15 minutes of
audio. A wrong number changes an answer and a wrong filler does not, so the
files with numeric questions are the ones worth the listening time. On
2026-09-17 the command chose:

| file | audio | questions with digits | example |
|---|---:|---:|---|
| conversation_sample_50 | 1:28 | 5 of 10 | Does the fasting blood sugar lie at 5.0 mmol/L? |
| conversation_sample_92 | 2:07 | 3 of 10 | Is the pulse recorded as 92? |
| conversation_sample_71 | 2:10 | 3 of 10 | Is the LDL cholesterol 4.2 mmol/L? |
| conversation_sample_42 | 2:44 | 3 of 10 | Was the blood pressure measured at 135/88? |
| conversation_sample_17 | 1:14 | 2 of 10 | Should the daily dose be 100 mg? |
| conversation_sample_75 | 1:36 | 2 of 10 | Does the patient deny any known exposure to COVID-19? |
| conversation_sample_55 | 1:44 | 2 of 10 | Did the HbA1c come out at 43 mmol/mol? |
| conversation_sample_6 | 1:29 | 1 of 10 | Will the patient take Fluconazole 50 mg? |

14.5 minutes of audio in total. `INDEX.md` suggests the first eight in
evaluator order instead (4, 5, 6, 10, 17, 18, 19, 20; 16.0 minutes, 4 numeric
questions in total against 21 here). Either set works for the scorer; this one
tests the numbers.

Starting from a better transcript than large-v3 is possible once a text-only
model has run (`canary-qwen-2.5b` has the best text of the models we run):

```
python bench/asr/compare.py --suggest-ref 8 --draft canary-qwen-2.5b
```

writes `bench/results/ref_drafts/<stem>.txt` in the same line format. Copy one
over the file here before correcting it.

## Setup, 5 minutes

- Open the file in a plain text editor.
- Open `data/audio/<stem>.mp3` in a player with keyboard seeking. In mpv the
  left and right arrows seek 5 seconds; in VLC, Shift plus left or right seeks
  3 seconds. Play at normal speed. Numbers go by too fast at 1.5x.

## Per file, about 4.5 minutes

Listen once from the start with the file beside the player and fix what is
wrong as you hear it. Rewind for anything with a number in it. Edit only the
words; leave the `[m:ss.s]` markers. In order of how much each kind of error
matters to the score:

1. Numbers. Doses, lab values, blood pressure, pulse, ages, years, durations,
   "twice a day". Write them as digits the way the models do: `50 mg`,
   `5.0 mmol/L`, `135/88`, `2017`, `HbA1c 47`. The scorer spells both sides out
   in words before comparing, so `5.0` and `five point zero` count as equal,
   and `135/88` matches `135 over 88` on the digits (the word "over" is then
   compared like any other word). Write the number you hear. If the speaker
   says "a hundred", write "a hundred", not `100`.
2. Drug names. Check the spelling when unsure (pro.medicin.dk or Wikipedia).
   The models produce plausible misspellings that are hard to catch by eye.
3. Units and abbreviations, as spoken. "milligrams" if said in full, `mg` if
   said as letters; "millimoles per litre" if said so, otherwise `mmol/L`.
   `HbA1c`, `LDL`, `BMI` and `COVID-19` as written here.
4. Words that flip an answer: no and know, increase and decrease, can and
   can't, "not".
5. Words nobody said, and words that were said and are missing. Whisper adds
   "Thank you." on trailing silence and drops one of two repeated words. If a
   whole passage is missing, add a line for it with an approximate marker.
6. Patient and doctor names, if you can hear the spelling.

Leave alone:

- Fillers and false starts ("um", "uh", "you know", "I, I think"). Whether
  they appear depends on the model's training, not on its accuracy, and we do
  not want to score that. Keep whatever the file has; add none, remove none.
- Punctuation, capitals, line breaks. The scorer removes them.
- Contractions. Keep the file's form ("don't") unless you hear the other
  form ("do not"). The scorer does not expand contractions, so a changed
  contraction costs two errors.
- British against American spelling, unless it is a different word.
- Inaudible stretches. Write nothing for them. Do not write "[inaudible]"; it
  would count as a word.
- Speaker labels. Do not add any.

When the file is done, insert `# checked` as its first line and save.

## Finish, 4 minutes

```
python bench/asr/compare.py
```

The first output line reads `ref: 8 checked of 39 file(s)` and the `WER ref`
column reads `(n=8)`. If the line also says `1 dropped as uncorrected model
text (conversation_sample_N)`, that file still equals a model's transcript
word for word; see above. Open `bench/results/asr/<tag>.json` and look at
`wer_detail.files`, the WER per file. A file above 10 percent usually has a
formatting problem (a speaker label in the middle of a line, a marker without
its closing bracket, a line pasted twice), not a bad model. A file at exactly
100 percent with `n_hyp: 0` is a transcript the model wrote no words for; the
scorer keeps it as all deletions, lists it under `empty_files` in the JSON and
shows `(n=8, 1 empty)` in the table, so a runner failure reads as one, not as
a better number. Commit `bench/ref/*.txt`; they are the only hand-made files
in the bench.

The scorer reads each file as UTF-8 text with any line breaks. It ignores
lines starting with `#` and strips a leading `[m:ss.s]` marker and a leading
speaker label written with a colon (`Doctor:`, `Patient:`, `Dr:`, `GP:`,
`D:`, `P:`) on a line. A hyphen is not a label separator: a line that starts
with `D-dimer`, `P-value` or `GP-led` is kept whole.
