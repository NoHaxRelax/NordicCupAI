"""How well can sentence-level ASR units reproduce the annotated evidence spans?

For every annotated yes question, build candidate units from the cached
transcript (Whisper segments; sentences split on punctuation using word times;
merges of 1..K consecutive sentences) and pick the candidate with the best tIoU.
That is the ceiling a perfect *selector* could reach with these timestamps. Also
report the start/end offsets of the best match, so a constant correction can be
estimated, and what a pad/shift does to the ceiling.

    python span_ceiling.py --model large-v3
"""
import argparse, csv, json, re, statistics as st
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE / 'data' / 'question_train.csv'
TR = HERE / 'transcripts'
END = re.compile(r'[.!?]$')


def tiou(g, p):
    inter = max(0.0, min(g[1], p[1]) - max(g[0], p[0]))
    union = max(g[1], p[1]) - min(g[0], p[0])
    return inter / union if union > 0 else 0.0


def sentences(data):
    """Split Whisper segments into sentence units on terminal punctuation."""
    out, cur = [], []
    for s in data['segments']:
        for w in s['words']:
            cur.append(w)
            if END.search(w['w'].strip()):
                out.append(cur); cur = []
        if cur:                      # segment boundary also ends a unit
            out.append(cur); cur = []
    return [(u[0]['start'], u[-1]['end'], ''.join(x['w'] for x in u).strip()) for u in out if u]


def merges(units, k):
    for i in range(len(units)):
        for j in range(i, min(len(units), i + k)):
            yield (units[i][0], units[j][1], j - i + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='large-v3')
    ap.add_argument('--k', type=int, default=4, help='max sentences merged')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CSV, encoding='utf-8')) if r['question_type'] == 'positive']
    cache = {}
    res = {'segment': [], 'sentence': [], f'merge<={a.k}': []}
    offs, best_n, lens = [], [], []
    missing = set()
    for r in rows:
        tid = r['transcript_id']
        f = TR / f'conversation_{tid}.{a.model}.json'
        if not f.exists():
            missing.add(tid); continue
        if tid not in cache:
            d = json.loads(f.read_text(encoding='utf-8'))
            sents = sentences(d)
            cache[tid] = (d, sents)
        d, sents = cache[tid]
        g = (float(r['evidence_start']), float(r['evidence_end']))
        seg_best = max((tiou(g, (s['start'], s['end'])) for s in d['segments']), default=0)
        sen_best = max((tiou(g, (u[0], u[1])) for u in sents), default=0)
        mb, mspan = 0, None
        for (s0, s1, n) in merges(sents, a.k):
            v = tiou(g, (s0, s1))
            if v > mb: mb, mspan = v, (s0, s1, n)
        res['segment'].append(seg_best); res['sentence'].append(sen_best); res[f'merge<={a.k}'].append(mb)
        if mspan:
            offs.append((g[0] - mspan[0], g[1] - mspan[1])); best_n.append(mspan[2]); lens.append(g[1] - g[0])
    if missing:
        print(f'({len(missing)} conversations not transcribed yet: skipped)')
    n = len(res['segment'])
    print(f'{n} annotated yes questions, model {a.model}\n')
    print('Oracle-selection tIoU ceiling (best candidate per gold span):')
    for k, v in res.items():
        v = sorted(v)
        print(f'  {k:<12} mean {st.mean(v):.3f}   median {v[len(v)//2]:.3f}   <0.5: {sum(x<0.5 for x in v)}/{n}')
    ds = sorted(o[0] for o in offs); de = sorted(o[1] for o in offs)
    med = lambda x: x[len(x)//2]
    print(f'\nOffset gold minus best-merge (positive = gold later):')
    print(f'  start: median {med(ds):+.2f}s  mean {st.mean(ds):+.2f}s  p25 {ds[len(ds)//4]:+.2f} p75 {ds[3*len(ds)//4]:+.2f}')
    print(f'  end:   median {med(de):+.2f}s  mean {st.mean(de):+.2f}s  p25 {de[len(de)//4]:+.2f} p75 {de[3*len(de)//4]:+.2f}')
    from collections import Counter
    print(f'  sentences in best merge: {sorted(Counter(best_n).items())}')
    # Constant correction: apply median offsets to every candidate and re-score
    ds_m, de_m = med(ds), med(de)
    corr = []
    for r in rows:
        tid = r['transcript_id']
        if tid not in cache: continue
        g = (float(r['evidence_start']), float(r['evidence_end']))
        corr.append(max(tiou(g, (s0 + ds_m, s1 + de_m)) for (s0, s1, _) in merges(cache[tid][1], a.k)))
    print(f'\nCeiling after shifting starts by {ds_m:+.2f}s and ends by {de_m:+.2f}s: mean {st.mean(corr):.3f}')
    # Worst cases
    worst = sorted(zip(res[f'merge<={a.k}'], rows), key=lambda x: x[0])[:8]
    print('\nWorst gold spans even with oracle selection:')
    for v, r in worst:
        print(f'  {v:.2f} {r["transcript_id"]:<10} [{float(r["evidence_start"]):.2f}-{float(r["evidence_end"]):.2f}] {r["question"]}')


if __name__ == '__main__':
    main()
