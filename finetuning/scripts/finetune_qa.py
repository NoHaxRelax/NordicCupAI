r"""Fine-tune deepset/deberta-v3-large-squad2 on the medical-appointment QA set.

Two folds, each holding out 10% of the *conversations* for test, with no
conversation appearing in more than one test fold. Every epoch the held-out fold
is scored the way the case scores: yes/no accuracy, mean temporal IoU of the
evidence spans, and the 0.4/0.6 combination of the two. Results go to a metrics
table and a four-panel plot.

    .\run.cmd python scripts/finetune_qa.py           # the real run, 5 epochs x 2 folds
    .\run.cmd python scripts/finetune_qa.py --smoke   # 2 min plumbing check

PowerShell needs the leading .\ ; cmd.exe does not.

Splitting is by conversation, never by row: ten questions share one context, so a
row-level split would put the same transcript on both sides of the wall.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent

# Same anchor convention the dataset was built with: the annotators' interval
# starts near the END of its first word. See data/DATASET.md.
START_OFFSET = -0.14
END_OFFSET = 0.12

# Case scoring weights.
ACCURACY_WEIGHT = 0.4
TIOU_WEIGHT = 0.6

# Categorical slots 1-3 of the reference palette; validated all-pairs, both modes.
FOLD_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#dedcd5'


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_data(data_path: Path, ctx_path: Path, keep_unaligned: bool):
    records = [json.loads(l) for l in data_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    contexts = json.loads(ctx_path.read_text(encoding='utf-8'))
    if not keep_unaligned:
        before = len(records)
        records = [r for r in records if r['is_impossible'] or r.get('alignment_ok', True)]
        dropped = before - len(records)
        if dropped:
            print(f'dropped {dropped} positives whose gold span is not recoverable from the transcript')
    return records, contexts


def make_folds(tids, n_folds: int, test_frac: float, seed: int):
    """Disjoint test sets: fold i takes the i-th block of the shuffled order."""
    order = sorted(tids)
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(order) * test_frac))
    if n_test >= len(order):
        raise SystemExit('need at least one training and one held-out conversation')
    if n_test * n_folds > len(order):
        raise SystemExit(f'cannot cut {n_folds} disjoint test sets of {n_test} from {len(order)} conversations')
    folds = []
    for i in range(n_folds):
        test = order[i * n_test:(i + 1) * n_test]
        folds.append(([t for t in order if t not in set(test)], test))
    return folds, n_test


def tiou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def chars_to_time(conv, cs: int, ce: int):
    """Predicted character span -> seconds, via the words it covers."""
    ws = [w for w in conv['words'] if w['char_end'] > cs and w['char_start'] < ce]
    if not ws:
        return None
    start = ws[0]['end'] + START_OFFSET
    end = ws[-1]['end'] + END_OFFSET
    dur = conv['duration']
    start = max(0.0, min(start, dur))
    end = max(start + 0.05, min(end, dur))
    return start, end


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def windows(tok, question, context, max_len, stride):
    """Char ranges that together cover the WHOLE context, each fitting in max_len.

    The obvious way to do this is return_overflowing_tokens + stride, but in
    transformers 5.17 that caps at two windows: a 6000-character context comes
    back as two windows covering 245 characters, and the rest of the text is
    silently dropped. So the chunking is done here, where full coverage can be
    asserted.
    """
    # An empty second string may be treated as a single sequence, omitting
    # the extra pair separator and silently truncating the last context token.
    overhead = (len(tok(question, add_special_tokens=False)['input_ids'])
                + tok.num_special_tokens_to_add(pair=True))
    budget = max_len - overhead
    if budget < 16:
        raise ValueError(f'question needs {overhead} of {max_len} tokens; no room for context')
    if stride < 0 or stride >= budget:
        raise ValueError(f'stride must be >= 0 and smaller than the context token budget ({budget})')
    offs = tok(context, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    n = len(offs)
    if n <= budget:
        return [(0, len(context))]
    step = max(1, budget - stride)
    out, i = [], 0
    while True:
        j = min(i + budget, n)
        out.append((offs[i][0], offs[j - 1][1]))
        if j >= n:
            return out
        i += step


def encode(tok, records, max_len, stride, target, training: bool):
    feats = collections.defaultdict(list)
    starts, ends, ex_ids, offsets = [], [], [], []
    a_start_key, a_end_key = ('answers', 'answer_end') if target == 'word' else ('answers_unit', 'answer_unit_end')

    for rec in records:
        question = rec['question'].lstrip()
        for c0, c1 in windows(tok, question, rec['context'], max_len, stride):
            enc = tok(question, rec['context'][c0:c1], truncation='only_second',
                      max_length=max_len, padding='max_length', return_offsets_mapping=True)
            seq_ids = enc.sequence_ids(0)
            # context offsets shifted back into whole-context coordinates
            offset = [(o[0] + c0, o[1] + c0) if seq_ids[k] == 1 else None
                      for k, o in enumerate(enc['offset_mapping'])]

            for key in ('input_ids', 'attention_mask', 'token_type_ids'):
                if key in enc:
                    feats[key].append(enc[key])
            ex_ids.append(rec['id'])

            if not training:
                offsets.append(offset)
                continue

            ctx_idx = [k for k, s in enumerate(seq_ids) if s == 1]
            if rec['is_impossible'] or not ctx_idx:
                starts.append(0)
                ends.append(0)
                continue
            start_char = rec[a_start_key]['answer_start'][0]
            end_char = rec[a_end_key]
            ts, te = ctx_idx[0], ctx_idx[-1]
            # answer not fully inside this window -> the window is a negative
            if offset[ts][0] > start_char or offset[te][1] < end_char:
                starts.append(0)
                ends.append(0)
                continue
            a = ts
            while a <= te and offset[a][1] <= start_char:
                a += 1
            b = te
            while b >= ts and offset[b][0] >= end_char:
                b -= 1
            starts.append(a)
            ends.append(b)

    feats = dict(feats)
    if training:
        feats['start_positions'] = starts
        feats['end_positions'] = ends
        return feats, None, None
    return feats, ex_ids, offsets


def postprocess(records, ex_ids, offsets, start_logits, end_logits, n_best, max_answer_tokens, null_threshold):
    """Standard SQuAD-v2 span selection: best non-null span vs the CLS score."""
    by_example = collections.defaultdict(list)
    for i, eid in enumerate(ex_ids):
        by_example[eid].append(i)
    preds = {}
    for rec in records:
        idxs = by_example.get(rec['id'], [])
        min_null, best = None, None
        for i in idxs:
            sl, el, off = start_logits[i], end_logits[i], offsets[i]
            null = sl[0] + el[0]
            if min_null is None or null < min_null:
                min_null = null
            s_idx = np.argsort(sl)[-1:-n_best - 1:-1]
            e_idx = np.argsort(el)[-1:-n_best - 1:-1]
            for s in s_idx:
                for e in e_idx:
                    if s >= len(off) or e >= len(off) or off[s] is None or off[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = sl[s] + el[e]
                    if best is None or score > best[0]:
                        best = (score, off[s][0], off[e][1])
        if best is None:
            preds[rec['id']] = (True, None)          # nothing extractable -> "no"
            continue
        score_diff = min_null - best[0]
        preds[rec['id']] = (score_diff > null_threshold, (best[1], best[2]))
    return preds


def detail(records, contexts, preds):
    """Per-question rows: what was predicted, where, and what it scored."""
    rows = []
    for rec in records:
        is_null, span = preds[rec['id']]
        said_yes = not is_null
        gold_yes = rec['label'] == 1
        times, t = None, 0.0
        if said_yes and span is not None:
            times = chars_to_time(contexts[rec['transcript_id']], span[0], span[1])
            if times and gold_yes:
                t = tiou((rec['evidence_start'], rec['evidence_end']), times)
        rows.append({
            'id': rec['id'],
            'transcript_id': rec['transcript_id'],
            'question_type': rec['question_type'],
            'gold_yes': gold_yes,
            'pred_yes': said_yes,
            'char_start': span[0] if span else None,
            'char_end': span[1] if span else None,
            'time_start': round(times[0], 3) if times else None,
            'time_end': round(times[1], 3) if times else None,
            'evidence_start': rec.get('evidence_start'),
            'evidence_end': rec.get('evidence_end'),
            'tiou': round(t, 4) if gold_yes else None,
        })
    return rows


def score(records, contexts, preds):
    """The case's own metric, on this fold."""
    rows = detail(records, contexts, preds)
    correct = sum(int(r['pred_yes'] == r['gold_yes']) for r in rows)
    tious = [r['tiou'] for r in rows if r['gold_yes']]
    diag = [r['tiou'] for r in rows if r['gold_yes'] and r['pred_yes']]
    acc = correct / len(records)
    mt = float(np.mean(tious)) if tious else 0.0
    return {
        'accuracy': acc,
        'mean_tiou': mt,
        'score': ACCURACY_WEIGHT * acc + TIOU_WEIGHT * mt,
        'tiou_when_yes': float(np.mean(diag)) if diag else 0.0,
        'n_pred_yes': sum(1 for r in records if not preds[r['id']][0]),
        'n_gold_yes': sum(1 for r in records if r['label'] == 1),
    }


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #

def n_batches(feats, bs):
    return math.ceil(len(feats['input_ids']) / bs)


def batches(feats, keys, bs, shuffle, device, seed=0):
    n = len(feats['input_ids'])
    idx = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, n, bs):
        chunk = idx[i:i + bs]
        yield {k: torch.tensor([feats[k][j] for j in chunk], device=device) for k in keys}


@torch.no_grad()
def infer(model, feats, bs, device, amp_dtype, quiet=False):
    model.eval()
    keys = [k for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in feats]
    starts, ends = [], []
    bar = tqdm(total=n_batches(feats, bs), desc='eval', unit='batch',
               leave=False, disable=quiet, position=1)
    for batch in batches(feats, keys, bs, False, device):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(**batch)
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
        bar.update(1)
    bar.close()
    return np.concatenate(starts), np.concatenate(ends)


def run_fold(fold_i, train_tids, test_tids, records, contexts, tok, args, device, amp_dtype):
    from transformers import AutoModelForQuestionAnswering, get_linear_schedule_with_warmup

    train_recs = [r for r in records if r['transcript_id'] in set(train_tids)]
    test_recs = [r for r in records if r['transcript_id'] in set(test_tids)]
    if args.smoke:
        train_recs = train_recs[:16]
    print(f'\n=== fold {fold_i + 1}: {len(train_tids)} train / {len(test_tids)} test conversations '
          f'({len(train_recs)} / {len(test_recs)} questions) ===')
    print(f'    test conversations: {", ".join(test_tids)}')

    tr_feats, _, _ = encode(tok, train_recs, args.max_len, args.stride, args.target, True)
    te_feats, te_ids, te_offsets = encode(tok, test_recs, args.max_len, args.stride, args.target, False)
    print(f'    {len(tr_feats["input_ids"])} train features, {len(te_feats["input_ids"])} test features')

    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.optim == 'adafactor':
        from transformers.optimization import Adafactor
        opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)
    elif args.optim == 'adamw8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    keys = [k for k in ('input_ids', 'attention_mask', 'token_type_ids',
                        'start_positions', 'end_positions') if k in tr_feats]
    batch_count = n_batches(tr_feats, args.batch)
    steps_per_epoch = math.ceil(batch_count / args.accum)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    history = []
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f'fold {fold_i + 1}/{args.folds}',
                     unit='epoch', disable=args.no_progress, position=0)
    for epoch in epoch_bar:
        model.train()
        t0, losses, step = time.time(), [], 0
        opt.zero_grad(set_to_none=True)
        step_bar = tqdm(total=n_batches(tr_feats, args.batch), desc=f'  epoch {epoch}/{args.epochs}',
                        unit='batch', leave=False, disable=args.no_progress, position=1)
        for bi, batch in enumerate(batches(tr_feats, keys, args.batch, True, device, seed=args.seed + epoch)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch).loss
            # The last accumulation group can be shorter than args.accum.
            # Weight batches by sample count, including a short final batch.
            group_first = (bi // args.accum) * args.accum * args.batch
            group_size = min(args.accum * args.batch, len(tr_feats['input_ids']) - group_first)
            batch_size = batch['input_ids'].shape[0]
            scaler.scale(loss * (batch_size / group_size)).backward()
            losses.append(loss.item())
            if (bi + 1) % args.accum == 0 or bi + 1 == batch_count:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            step_bar.update(1)
            step_bar.set_postfix(loss=f'{np.mean(losses[-20:]):.3f}', opt_steps=step)
        step_bar.close()

        sl, el = infer(model, te_feats, args.eval_batch, device, amp_dtype, quiet=args.no_progress)
        preds = postprocess(test_recs, te_ids, te_offsets, sl, el,
                            args.n_best, args.max_answer_tokens, args.null_threshold)
        m = score(test_recs, contexts, preds)
        m.update(epoch=epoch, fold=fold_i + 1, train_loss=float(np.mean(losses)),
                 seconds=round(time.time() - t0, 1))
        history.append(m)
        epoch_bar.set_postfix(acc=f'{m["accuracy"]:.3f}', tIoU=f'{m["mean_tiou"]:.3f}',
                              score=f'{m["score"]:.3f}')
        tqdm.write(f'    epoch {epoch:>2}  loss {m["train_loss"]:.4f}  acc {m["accuracy"]:.3f}  '
                   f'tIoU {m["mean_tiou"]:.3f}  score {m["score"]:.3f}  '
                   f'(yes {m["n_pred_yes"]}/{m["n_gold_yes"]})  {m["seconds"]:.0f}s')

        # final epoch: keep the per-question predictions so a committee can be
        # assembled from several models' held-out answers afterwards
        if args.dump_preds and epoch == args.epochs:
            rows = detail(test_recs, contexts, preds)
            for r in rows:
                r['model'] = args.model
                r['fold'] = fold_i + 1
                r['epoch'] = epoch
            with open(Path(args.out_dir) / 'preds.jsonl', 'a', encoding='utf-8', newline='\n') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    epoch_bar.close()

    if args.save_dir:
        out = Path(args.save_dir) / f'fold{fold_i + 1}'
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f'    saved -> {out}')

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return history


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def plot(history, out_dir: Path, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folds = sorted({h['fold'] for h in history})
    epochs = sorted({h['epoch'] for h in history})
    panels = [
        ('accuracy', 'Yes/no accuracy on held-out fold', (0, 1)),
        ('mean_tiou', 'Mean temporal IoU of evidence spans', (0, 1)),
        ('score', f'Case score  ({ACCURACY_WEIGHT}×acc + {TIOU_WEIGHT}×tIoU)', (0, 1)),
        ('train_loss', 'Training loss', None),
    ]

    def place_end_labels(ax, items):
        """Direct labels at the line ends, pushed apart so they never overlap."""
        lo, hi = ax.get_ylim()
        gap = 0.058 * (hi - lo)
        items = sorted(items, key=lambda t: t[0])
        ys = [it[0] for it in items]
        for i in range(1, len(ys)):                       # spread upward
            ys[i] = max(ys[i], ys[i - 1] + gap)
        over = ys[-1] - (hi - 0.02 * (hi - lo))
        if over > 0:                                      # then re-seat inside the axes
            ys = [y - over for y in ys]
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        for (_, x, text, color, weight), y in zip(items, ys):
            ax.annotate(text, xy=(x, y), xytext=(7, 0), textcoords='offset points',
                        color=color, fontsize=9, va='center', fontweight=weight,
                        annotation_clip=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), facecolor='#fcfcfb')
    for ax, (key, title, ylim) in zip(axes.ravel(), panels):
        ax.set_facecolor('#fcfcfb')
        labels = []
        for f in folds:
            rows = [h for h in history if h['fold'] == f]
            xs = [h['epoch'] for h in rows]
            ys = [h[key] for h in rows]
            c = FOLD_COLORS[(f - 1) % len(FOLD_COLORS)]
            ax.plot(xs, ys, color=c, linewidth=2, marker='o', markersize=5,
                    markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=f'fold {f}', zorder=3)
            if ys:  # relief rule: identity is never carried by colour alone
                labels.append((ys[-1], xs[-1], f'fold {f}', c, 'normal'))
        mean = [float(np.mean([h[key] for h in history if h['epoch'] == e])) for e in epochs]
        ax.plot(epochs, mean, color=INK, linewidth=2.5, linestyle='--', label='mean', zorder=4)
        labels.append((mean[-1], epochs[-1], 'mean', INK, 'bold'))

        ax.set_title(title, color=INK, fontsize=11, loc='left', pad=10)
        ax.set_xlabel('epoch', color=INK_SOFT, fontsize=9)
        ax.set_xticks(epochs)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.22 * (epochs[-1] - epochs[0] + 1))
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9)
        place_end_labels(ax, labels)

    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc='lower right')
    fig.suptitle(f'{args.model}  —  {args.epochs} epochs, {len(folds)} disjoint test folds',
                 color=INK, fontsize=13, x=0.02, ha='left', y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / 'metrics.png'
    fig.savefig(path, dpi=150, facecolor='#fcfcfb')
    print(f'plot -> {path}')


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'qa_train.jsonl')
    p.add_argument('--contexts', type=Path, default=ROOT / 'data' / 'contexts.json')
    p.add_argument('--model', default='deepset/deberta-v3-large-squad2')
    p.add_argument('--out', type=Path, default=None, help='run directory (default runs/<timestamp>)')
    p.add_argument('--save-dir', default=None, help='also save the fine-tuned weights here')

    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--folds', type=int, default=2)
    p.add_argument('--test-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=20260918)

    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--optim', choices=['adafactor', 'adamw', 'adamw8bit'], default='adafactor')
    p.add_argument('--no-grad-checkpointing', dest='grad_checkpointing', action='store_false')

    p.add_argument('--max-len', type=int, default=384)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--target', choices=['word', 'unit'], default='word')
    p.add_argument('--n-best', type=int, default=20)
    p.add_argument('--max-answer-tokens', type=int, default=80)
    p.add_argument('--null-threshold', type=float, default=0.0)

    p.add_argument('--keep-unaligned', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--fp16', action='store_true', help='use fp16 instead of bf16')
    p.add_argument('--smoke', action='store_true', help='1 fold, 1 epoch, 16 examples')
    p.add_argument('--no-progress', action='store_true', help='no tqdm bars (for logs and CI)')
    p.add_argument('--dump-preds', action='store_true',
                   help='write per-question held-out predictions to preds.jsonl (for committee analysis)')
    p.add_argument('--only-folds', nargs='+', type=int,
                   help='run only these 1-based fold numbers; the split is unchanged, '
                        'use a fresh --out directory for a partial rerun')
    args = p.parse_args()

    if args.smoke:
        args.folds, args.epochs = 1, 1

    for name in ('epochs', 'folds', 'batch', 'accum', 'eval_batch', 'max_len', 'n_best', 'max_answer_tokens'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if not 0 < args.test_frac < 1 or args.stride < 0 or not 0 <= args.warmup < 1:
        p.error('Require 0 < --test-frac < 1, --stride >= 0 and 0 <= --warmup < 1')
    if args.only_folds and any(i < 1 or i > args.folds for i in args.only_folds):
        p.error('--only-folds must contain numbers from 1 through --folds')

    out_dir = args.out or ROOT / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    if out_dir.exists() and any(out_dir.iterdir()):
        p.error('Output directory is not empty; choose a fresh --out to preserve existing results')

    from transformers import AutoTokenizer

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    amp_dtype = None
    if device.type == 'cuda':
        amp_dtype = torch.float16 if args.fp16 else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f'device: {device}' + (f' ({torch.cuda.get_device_name(0)}, {amp_dtype})' if device.type == 'cuda' else ''))
    if device.type == 'cpu':
        print('WARNING: no CUDA. deberta-v3-large on CPU is many hours per fold. '
              'Install a CUDA torch build, or use --smoke to check the plumbing.')

    records, contexts = load_data(args.data, args.contexts, args.keep_unaligned)
    tids = sorted({r['transcript_id'] for r in records})
    folds, n_test = make_folds(tids, args.folds, args.test_frac, args.seed)

    held_out = [t for _, te in folds for t in te]
    assert len(held_out) == len(set(held_out)), 'test folds overlap'
    print(f'{len(records)} questions, {len(tids)} conversations')
    print(f'{args.folds} folds x {n_test} test conversations = {len(held_out)} held out, no overlap')

    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir
    if args.dump_preds:                       # append-only across folds; start clean
        (out_dir / 'preds.jsonl').unlink(missing_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tok.is_fast:
        raise SystemExit('need a fast tokenizer for offset mapping; pip install sentencepiece protobuf')

    # Fold numbering stays tied to the split, so --only-folds 3 reruns exactly
    # the fold a crash lost, and its rows merge back with the others.
    todo = list(enumerate(folds))
    if args.only_folds:
        keep = {int(x) for x in args.only_folds}
        todo = [(i, f) for i, f in todo if i + 1 in keep]
        print(f'running only fold(s) {sorted(keep)} of {len(folds)}')

    history = []
    for i, (train_tids, test_tids) in todo:
        history += run_fold(i, train_tids, test_tids, records, contexts, tok, args, device, amp_dtype)
        (out_dir / 'metrics.json').write_text(json.dumps(history, indent=1), encoding='utf-8')

    cols = ['fold', 'epoch', 'train_loss', 'accuracy', 'mean_tiou', 'score',
            'tiou_when_yes', 'n_pred_yes', 'n_gold_yes', 'seconds']
    with open(out_dir / 'metrics.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for h in history:
            w.writerow({c: h[c] for c in cols})

    (out_dir / 'config.json').write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1), encoding='utf-8')

    print(f'\nBest epoch by mean case score across folds:')
    for e in sorted({h['epoch'] for h in history}):
        rows = [h for h in history if h['epoch'] == e]
        print(f'  epoch {e:>2}  acc {np.mean([r["accuracy"] for r in rows]):.3f}  '
              f'tIoU {np.mean([r["mean_tiou"] for r in rows]):.3f}  '
              f'score {np.mean([r["score"] for r in rows]):.3f}')

    plot(history, out_dir, args)
    print(f'metrics -> {out_dir / "metrics.csv"}')


if __name__ == '__main__':
    raise SystemExit(main())
