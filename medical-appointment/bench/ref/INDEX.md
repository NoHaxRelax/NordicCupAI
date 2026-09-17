# Reference transcripts: how to correct them

One file per conversation, pre-filled with Whisper large-v3. Each line is one segment: a `[m:ss.s]` marker for finding the spot in the MP3, then the text. The marker is ignored by the scorer; edit only the words.

Rules:

- Play `data/audio/<name>.mp3` in any player and read along. Fix every word that is wrong, especially numbers, units, doses, drug names, dates and body parts.
- Keep what was said as spoken. Fillers (`um`, `okay`, `yeah`) stay if spoken; do not tidy grammar.
- Digits or words: write what is natural (`100 mg`, `two weeks`). The scorer spells digits out on both sides before comparing.
- Speaker tag after the marker: `[D]` doctor, `[P]` patient, e.g. `[0:01.9] [D] Christian, come in.` Two consecutive lines from the same speaker are normal; tag both. If one line holds both speakers, split it into two lines and copy the marker onto both. The scorer strips the tag; it is used for role experiments and turn boundaries.
- Do not merge lines.
- If a whole passage is missing, add a line for it with an approximate marker.
- When a file is done, add `# checked` as its first line.

Suggested eight (from `python bench/asr/compare.py --suggest-ref 8`, the digit-heavy files; see PROTOCOL.md): sample_50, sample_92, sample_71, sample_42, sample_55, sample_17, sample_75, sample_6. Samples 4 and 5 already done count too.

| conversation | seconds | lines | questions with digits |
|---|---|---|---|
| sample_4 * | 106 | 28 | 0 |
| sample_5 * | 169 | 52 | 1 |
| sample_6 * | 89 | 24 | 1 |
| sample_10 | 102 | 32 | 0 |
| sample_17 * | 74 | 21 | 2 |
| sample_18 | 94 | 31 | 0 |
| sample_19 | 95 | 29 | 0 |
| sample_20 | 232 | 39 | 0 |
| sample_23 | 100 | 23 | 1 |
| sample_33 | 102 | 26 | 0 |
| sample_37 | 101 | 30 | 0 |
| sample_39 | 83 | 19 | 0 |
| sample_42 * | 164 | 41 | 3 |
| sample_43 | 98 | 33 | 0 |
| sample_47 | 139 | 42 | 0 |
| sample_48 | 121 | 50 | 0 |
| sample_50 * | 88 | 24 | 5 |
| sample_52 | 136 | 34 | 0 |
| sample_54 | 116 | 38 | 0 |
| sample_55 * | 104 | 31 | 2 |
| sample_56 | 106 | 39 | 1 |
| sample_57 | 172 | 44 | 0 |
| sample_63 | 89 | 25 | 1 |
| sample_64 | 134 | 32 | 0 |
| sample_66 | 153 | 40 | 0 |
| sample_67 | 112 | 45 | 1 |
| sample_69 | 84 | 25 | 0 |
| sample_70 | 130 | 36 | 0 |
| sample_71 * | 130 | 32 | 3 |
| sample_75 * | 96 | 46 | 2 |
| sample_77 | 103 | 23 | 0 |
| sample_79 | 222 | 41 | 0 |
| sample_81 | 137 | 28 | 0 |
| sample_82 | 98 | 37 | 0 |
| sample_84 | 228 | 43 | 2 |
| sample_86 | 156 | 30 | 0 |
| sample_90 | 80 | 22 | 0 |
| sample_92 * | 127 | 42 | 3 |
| sample_95 | 96 | 29 | 0 |
