"""Zero-shot SQuAD2 extractive QA over the 39 turbo-rebuilt conversations.

Runs an off-the-shelf extractive model over every one of the 390 questions and
records ONE span per question -- including the questions it would call
unanswerable, because the competition metric credits a span even on a "no"
answer.  The null (CLS) score is recorded alongside so a later stage can
threshold instead of re-running.

Windowing: contexts run to ~3.3k characters, past the model's 512-token limit,
so each (question, context) pair is tokenised with a sliding window
(max_length 384, doc_stride 128).  The best non-null span is taken across all
windows of a pair; the null score is the MINIMUM across windows, the standard
squad_v2 convention.  Token indices are mapped back to characters in the FULL
context through the tokenizer's offset mapping, then to seconds by the dataset
builder's own rule, guarded at the word boundaries (see spanlib).

Usage:
    python run_zeroshot.py --model deepset/roberta-base-squad2 --tag roberta-base
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(HERE / 'hf'))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from spanlib import chars_to_time

sys.stdout.reconfigure(encoding='utf-8')

MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 60      # gold answers reach 167 chars (~45 tokens); not binding
NBEST = 20
BATCH = 32


# The fast tokenizer's own return_overflowing_tokens path does NOT window a
# long context here: it truncates the context to max_length tokens FIRST and
# only overflows within that, so a 779-token context is silently cut at ~384
# (verified on sample_79: two windows covering 384 unique context tokens, the
# tail discarded). Windows are therefore cut by hand, in token space, and
# expanded outward to whitespace so no window starts or ends mid-word.
WINDOW_MARGIN = 16          # slack absorbed by the outward expansion


def window_char_ranges(context, offs, capacity):
    """Character ranges of the sliding windows over one context."""
    n = len(offs)
    W = max(1, capacity - WINDOW_MARGIN)
    step = max(1, W - DOC_STRIDE)
    ranges, i = [], 0
    while True:
        j = min(n, i + W)
        a, b = offs[i][0], offs[j - 1][1]
        while a > 0 and not context[a - 1].isspace():
            a -= 1
        while b < len(context) and not context[b].isspace():
            b += 1
        if not ranges or (a, b) != ranges[-1]:
            ranges.append((a, b))
        if j >= n:
            break
        i += step
    return ranges


def build_features(tok, records, ctx_offsets):
    """One flat list of windows over all records, each tagged with its record."""
    n_special = tok.num_special_tokens_to_add(pair=True)
    feats, n_trunc = [], 0
    for ri, r in enumerate(records):
        context = r['context']
        offs = ctx_offsets[r['transcript_id']]
        q_len = len(tok(r['question'], add_special_tokens=False)['input_ids'])
        capacity = MAX_LENGTH - q_len - n_special
        for a, b in window_char_ranges(context, offs, capacity):
            enc = tok(r['question'], context[a:b], max_length=MAX_LENGTH,
                      truncation='only_second', return_offsets_mapping=True,
                      padding='max_length')
            seq_ids = enc.sequence_ids(0)
            assert seq_ids is not None, 'need a fast tokenizer for sequence_ids'
            # keep offsets only for context tokens, shifted back to the FULL
            # context; everything else is invalid
            offsets = [(o[0] + a, o[1] + a) if seq_ids[k] == 1 else None
                       for k, o in enumerate(enc['offset_mapping'])]
            reach = max((o[1] for o in offsets if o), default=a)
            if reach < a + len(context[a:b].rstrip()):
                n_trunc += 1        # window itself got truncated: should not happen
            feats.append({
                'rec': ri,
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'offsets': offsets,
            })
    if n_trunc:
        print(f'WARNING: {n_trunc} windows were truncated inside their own char range')
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepset/roberta-base-squad2')
    ap.add_argument('--tag', default='roberta-base')
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    a = ap.parse_args()

    t_wall0 = time.perf_counter()

    records = [json.loads(l) for l in io.open(a.data / 'qa_train.jsonl', encoding='utf-8')]
    ctx = json.load(open(a.data / 'contexts.json', encoding='utf-8'))
    print(f'{len(records)} questions over {len(ctx)} conversations')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForQuestionAnswering.from_pretrained(a.model).to(device).eval()
    t_load = time.perf_counter() - t0
    print(f'model {a.model} on {device}, loaded in {t_load:.2f}s')

    t0 = time.perf_counter()
    ctx_offsets = {t: tok(c['context'], add_special_tokens=False,
                          return_offsets_mapping=True)['offset_mapping']
                   for t, c in ctx.items()}
    feats = build_features(tok, records, ctx_offsets)
    t_feat = time.perf_counter() - t0
    per_rec = np.bincount([f['rec'] for f in feats], minlength=len(records))
    print(f'{len(feats)} windows ({per_rec.min()}-{per_rec.max()} per question, '
          f'mean {per_rec.mean():.2f}), tokenised in {t_feat:.2f}s')

    # self-check: every context must be covered end to end, with no gap
    by_rec_chk = {}
    for f in feats:
        by_rec_chk.setdefault(f['rec'], []).extend(
            [o for o in f['offsets'] if o is not None])
    n_gap = 0
    for ri, r in enumerate(records):
        covered, reach = True, 0
        for s, e in sorted(by_rec_chk[ri]):
            if s > reach + 1:
                covered = False
                break
            reach = max(reach, e)
        if not covered or reach < len(r['context'].rstrip()):
            n_gap += 1
    print(f'coverage self-check: {n_gap} of {len(records)} questions have a gap '
          f'or short reach')
    if n_gap:
        raise SystemExit('windowing does not cover the full context - refusing to score')

    # ---- forward passes ----------------------------------------------------
    t0 = time.perf_counter()
    all_start, all_end = [], []
    with torch.no_grad():
        for i in range(0, len(feats), BATCH):
            chunk = feats[i:i + BATCH]
            ids = torch.tensor([f['input_ids'] for f in chunk], device=device)
            am = torch.tensor([f['attention_mask'] for f in chunk], device=device)
            out = model(input_ids=ids, attention_mask=am)
            all_start.append(out.start_logits.float().cpu().numpy())
            all_end.append(out.end_logits.float().cpu().numpy())
    if device == 'cuda':
        torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    start_logits = np.concatenate(all_start)
    end_logits = np.concatenate(all_end)
    print(f'forward passes: {t_fwd:.2f}s ({1000 * t_fwd / len(feats):.1f} ms/window)')
    if device == 'cuda':
        print(f'peak GPU: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB allocated')

    # ---- per-record postprocessing ----------------------------------------
    by_rec = {}
    for fi, f in enumerate(feats):
        by_rec.setdefault(f['rec'], []).append(fi)

    out = {}
    n_null_win = 0
    for ri, r in enumerate(records):
        context = r['context']
        words = ctx[r['transcript_id']]['words']
        best = None            # (score, cs, ce)
        null_score = None      # min over windows, squad_v2 convention

        for fi in by_rec[ri]:
            sl, el = start_logits[fi], end_logits[fi]
            offsets = feats[fi]['offsets']
            ns = float(sl[0] + el[0])         # CLS
            null_score = ns if null_score is None else min(null_score, ns)

            s_idx = np.argsort(sl)[-1:-NBEST - 1:-1]
            e_idx = np.argsort(el)[-1:-NBEST - 1:-1]
            for si in s_idx:
                if offsets[si] is None:
                    continue
                for ei in e_idx:
                    if offsets[ei] is None or ei < si or ei - si + 1 > MAX_ANSWER_TOKENS:
                        continue
                    sc = float(sl[si] + el[ei])
                    if best is None or sc > best[0]:
                        best = (sc, int(offsets[si][0]), int(offsets[ei][1]))

        if best is None:       # no valid context pair anywhere: should not happen
            n_null_win += 1
            best = (float('-inf'), 0, len(context))

        score, raw_cs, raw_ce = best
        conv = chars_to_time(words, raw_cs, raw_ce)
        cs, ce, ts, te = conv
        out[r['id']] = {
            'char_start': cs,
            'char_end': ce,
            'text': context[cs:ce],
            'time_start': round(ts, 3),
            'time_end': round(te, 3),
            'null_score': round(null_score, 4),
            'best_span_score': round(score, 4),
            'raw_char_start': raw_cs,
            'raw_char_end': raw_ce,
            'raw_text': context[raw_cs:raw_ce],
            'transcript_id': r['transcript_id'],
            'n_windows': len(by_rec[ri]),
        }

    if n_null_win:
        print(f'WARNING: {n_null_win} questions had no valid span in any window')

    path = HERE / f'spans_{a.tag}.json'
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    t_wall = time.perf_counter() - t_wall0
    print(f'\nwrote {path}  ({len(out)} questions)')
    print(f'wall time {t_wall:.1f}s  (load {t_load:.1f}, tokenise {t_feat:.1f}, forward {t_fwd:.1f})')
    json.dump({'wall_s': round(t_wall, 2), 'load_s': round(t_load, 2),
               'tokenise_s': round(t_feat, 2), 'forward_s': round(t_fwd, 2),
               'n_windows': len(feats), 'model': a.model, 'device': device,
               'max_length': MAX_LENGTH, 'doc_stride': DOC_STRIDE,
               'max_answer_tokens': MAX_ANSWER_TOKENS},
              open(HERE / f'timing_{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
