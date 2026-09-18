r"""Train several extractive-QA architectures on the same folds, for the committee test.

Each model is fine-tuned separately with identical folds and hyper-parameters, and
its held-out predictions are dumped. scripts/committee_eval.py then asks whether
combining them beats the best single one.

    .\run.cmd python scripts/run_committee.py --epochs 2 --folds 3

Members are chosen for *disagreement*, not just strength: three pretraining
objectives (disentangled-attention MLM, plain MLM, whole-word-masked MLM) over
three tokenizers (SentencePiece, byte-BPE, WordPiece). All ship safetensors,
which transformers 5 needs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MEMBERS = [
    ('deberta-v3-large', 'deepset/deberta-v3-large-squad2'),
    ('roberta-large', 'deepset/roberta-large-squad2'),
    ('bert-large-wwm', 'deepset/bert-large-uncased-whole-word-masking-squad2'),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--out', type=Path, default=ROOT / 'runs' / 'committee')
    p.add_argument('--only', nargs='+', choices=[m[0] for m in MEMBERS],
                   help='train just these members, by slug')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                   help='everything after this is passed through to finetune_qa.py')
    a = p.parse_args()

    members = [m for m in MEMBERS if not a.only or m[0] in a.only]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'{len(members)} members x {a.folds} folds x {a.epochs} epochs -> {a.out}\n')

    results = []
    for i, (slug, model_id) in enumerate(members, 1):
        out = a.out / slug
        print(f'{"=" * 70}\n[{i}/{len(members)}] {slug}  ({model_id})\n{"=" * 70}')
        cmd = [sys.executable, str(ROOT / 'scripts' / 'finetune_qa.py'),
               '--model', model_id, '--epochs', str(a.epochs), '--folds', str(a.folds),
               '--out', str(out), '--dump-preds'] + a.extra
        t0 = time.time()
        rc = subprocess.call(cmd)
        mins = (time.time() - t0) / 60
        results.append((slug, rc, mins))
        print(f'\n[{i}/{len(members)}] {slug}: {"ok" if rc == 0 else f"FAILED rc={rc}"} in {mins:.1f} min\n')

    print('=' * 70)
    for slug, rc, mins in results:
        print(f'  {slug:<20} {"ok" if rc == 0 else f"FAILED rc={rc}":<14} {mins:>5.1f} min')
    ok = [s for s, rc, _ in results if rc == 0]
    if len(ok) < 2:
        print('\nfewer than two members trained; nothing to combine')
        return 0 if len(ok) == len(members) else 1
    print(f'\nNow: .\\run.cmd python scripts/committee_eval.py --runs {a.out}')
    return 0 if len(ok) == len(members) else 1


if __name__ == '__main__':
    raise SystemExit(main())
