"""Print the 19 dumped validation conversations: the ten questions with the served
model's answers, then the large-v3-turbo transcript as numbered, timed segments.
This is the view the hand answers in agent_answers.md refer to (#nn = segment index).

    python bench/mine/dump_val.py            # all
    python bench/mine/dump_val.py 3 80       # only these samples
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parents[2]
D = CASE / 'request_dump'


def main(argv: list[str]) -> int:
    ans = {}
    for line in (D / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        d = json.loads(line); ans[Path(d['file']).stem] = d
    stems = sorted(ans, key=lambda s: int(re.sub(r'\D', '', s)))
    if argv:
        want = {f'conversation_sample_{int(x)}' for x in argv}
        stems = [s for s in stems if s in want]
    for s in stems:
        d = ans[s]
        tf = next((D / 'transcripts').glob(f'{s}.*.json'), None)
        if tf is None:
            print(f'\n######## {s}  (no transcript yet)'); continue
        t = json.loads(tf.read_text(encoding='utf-8'))
        print(f'\n######## {s}  ({t["segments"][-1]["end"]:.1f}s, {len(t["segments"])} segs, {t.get("model")})')
        for i, (q, a) in enumerate(zip(d['questions'], d['answers'])):
            print(f'  Q{i + 1} [model {"yes" if a else "no"}] {q}')
        for i, seg in enumerate(t['segments']):
            print(f'  #{i:02d} {seg["start"]:6.2f}-{seg["end"]:6.2f} {seg["text"].strip()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
