"""Turn the hand labels from the /label page (bench/mine/val_labels/*.json)
into the probe endpoint's table (bench/mine/current_answers.json): per file a
list of {answer, start, end}. Questions without a label fall back to the
model's answer from request_dump/answers.jsonl (null span). Prints coverage.

    python bench/mine/labels_to_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
LABELS = HERE / 'val_labels'
DUMP = CASE / 'request_dump' / 'answers.jsonl'
OUT = HERE / 'current_answers.json'


def main():
    model = {}
    if DUMP.exists():
        for line in DUMP.read_text(encoding='utf-8').splitlines():
            d = json.loads(line); model[d['file']] = d
    table, n_lab, n_span, n_q = {}, 0, 0, 0
    for stem, d in sorted(model.items()):
        lab = LABELS / f'{Path(stem).stem}.json'
        items = json.loads(lab.read_text(encoding='utf-8')).get('items', []) if lab.exists() else []
        row = []
        for i, q in enumerate(d['questions']):
            n_q += 1
            it = items[i] if i < len(items) else {}
            if it.get('answer') is not None:
                n_lab += 1
                ok = it['answer'] and it.get('start') is not None and it.get('end') is not None
                n_span += int(bool(ok))
                row.append({'answer': bool(it['answer']), 'start': it.get('start') if ok else None, 'end': it.get('end') if ok else None})
            else:
                row.append({'answer': bool(d['answers'][i]), 'start': None, 'end': None})
        table[stem] = row
    OUT.write_text(json.dumps(table, indent=1), encoding='utf-8')
    print(f'{len(table)} files, {n_q} questions: {n_lab} hand-labelled ({n_span} with a span), {n_q - n_lab} from the model; wrote {OUT}')


if __name__ == '__main__':
    main()
