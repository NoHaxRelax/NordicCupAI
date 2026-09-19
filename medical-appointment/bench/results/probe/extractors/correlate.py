"""Correlation / routing analysis: incumbent vs zero-shot extractors.

Everything is recomputed from the raw span files; stored tiou_*.json is used
only as a cross-check.
"""
import json, math, sys
import numpy as np

BASE = r"C:\Users\edlun\AppData\Local\Temp\claude\c--Users-edlun-Desktop-lucky-shots-NordicCupAI\bf48d8ae-2b16-4039-be3b-c570527b37ee\scratchpad\probe"
ACC = 389 / 390  # incumbent accuracy, unchanged by any span routing


def tiou(a, b):
    if a is None or b is None:
        return 0.0
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    return float(xc @ yc / math.sqrt((xc @ xc) * (yc @ yc)))


def rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def perm_p(x, y, stat=pearson, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = stat(x, y)
    y = np.asarray(y, float)
    cnt = sum(1 for _ in range(n) if abs(stat(x, rng.permutation(y))) >= abs(obs) - 1e-12)
    return (cnt + 1) / (n + 1)


inc = json.load(open(f"{BASE}\\incumbent.json", encoding="utf-8"))
QIDS = [k for k, v in inc.items() if v["gold"] is not None]
QIDS.sort()
assert len(QIDS) == 195
GOLD = {q: inc[q]["gold"] for q in QIDS}
CONV = {q: inc[q]["transcript_id"] for q in QIDS}
inc_t = np.array([tiou(inc[q]["span"], GOLD[q]) for q in QIDS])

# cross-check against stored incumbent tiou
stored = np.array([inc[q]["tiou"] for q in QIDS])
assert np.max(np.abs(stored - inc_t)) < 1e-9, np.max(np.abs(stored - inc_t))

# question metadata (type) from the turbo build
meta = {}
for line in open(f"{BASE}\\data\\qa_train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    meta[r["id"]] = r

MODELS = [("deberta-large", "spans_deberta-large.json"),
          ("roberta-base", "spans_roberta-base.json")]

out = {}
for tag, fn in MODELS:
    sp = json.load(open(f"{BASE}\\{fn}", encoding="utf-8"))
    ext_span = {q: [sp[q]["time_start"], sp[q]["time_end"]] for q in QIDS}
    t = np.array([tiou(ext_span[q], GOLD[q]) for q in QIDS])
    # cross-check vs stored tiou file
    st = json.load(open(f"{BASE}\\tiou_{tag}.json", encoding="utf-8"))
    d = max(abs(st[q] - t[i]) for i, q in enumerate(QIDS))
    margin = np.array([sp[q]["best_span_score"] - sp[q]["null_score"] for q in QIDS])
    agree = np.array([tiou(inc[q]["span"], ext_span[q]) for q in QIDS])
    out[tag] = dict(spans=sp, ext_span=ext_span, t=t, margin=margin, agree=agree,
                    stored_maxdiff=d)

R = []
def p(s=""):
    R.append(s); print(s)

p("# Probe analysis: incumbent vs zero-shot extractors")
p()
p(f"Incumbent mean tIoU (recomputed from incumbent.json spans vs gold) = {inc_t.mean():.4f}")
p(f"Incumbent score 0.4*{ACC:.6f} + 0.6*{inc_t.mean():.6f} = {0.4*ACC + 0.6*inc_t.mean():.6f}")
p()

for tag, _ in MODELS:
    o = out[tag]
    t = o["t"]
    p(f"## {tag}")
    p(f"stored-vs-recomputed tIoU max abs diff = {o['stored_maxdiff']:.2e}")
    p(f"(a) mean tIoU = {t.mean():.4f}   median {np.median(t):.4f}   "
      f"zeros {int((t==0).sum())}   >=0.6 {int((t>=0.6).sum())}")
    pr, sr = pearson(inc_t, t), spearman(inc_t, t)
    p(f"(b) Pearson r = {pr:.4f}  (perm p = {perm_p(inc_t, t, pearson):.4f})")
    p(f"    Spearman rho = {sr:.4f}  (perm p = {perm_p(inc_t, t, spearman):.4f})")
    ig = inc_t >= 0.6; eg = t >= 0.6
    p(f"    2x2 at 0.6: both good {int((ig&eg).sum())}, inc good/ext bad {int((ig&~eg).sum())}, "
      f"inc bad/ext good {int((~ig&eg).sum())}, both bad {int((~ig&~eg).sum())}")
    # phi coefficient
    p(f"    phi (binary corr at 0.6) = {pearson(ig.astype(float), eg.astype(float)):.4f}")
    resc = int(((inc_t < 0.3) & (t >= 0.6)).sum())
    lost = int(((t < 0.3) & (inc_t >= 0.6)).sum())
    p(f"    rescuable (inc<0.3 & ext>=0.6) = {resc}    reverse (ext<0.3 & inc>=0.6) = {lost}")
    p(f"    inc<0.3 count = {int((inc_t<0.3).sum())}, ext<0.3 count = {int((t<0.3).sum())}")
    mx = np.maximum(inc_t, t); mn = np.minimum(inc_t, t)
    sc_or = 0.4*ACC + 0.6*mx.mean()
    p(f"(c) oracle-of-two mean tIoU = {mx.mean():.4f}   score = {sc_or:.4f}   "
      f"gain +{sc_or - (0.4*ACC + 0.6*inc_t.mean()):.4f}")
    p(f"    worst-of-two mean tIoU = {mn.mean():.4f}")
    p(f"    ext strictly better on {int((t>inc_t).sum())} / 195 questions")
    p("(d) routing curve  p -> score")
    for pp in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m = pp*mx.mean() + (1-pp)*mn.mean()
        p(f"    p={pp:.1f}  mean tIoU {m:.4f}  score {0.4*ACC + 0.6*m:.4f}")
    be = (inc_t.mean() - mn.mean()) / (mx.mean() - mn.mean())
    p(f"    break-even p = {be:.4f}")
    p()
    o.update(mx=mx, mn=mn, breakeven=be, oracle=sc_or, pearson=pr, spearman=sr)

json.dump({t: dict(mean=float(out[t]["t"].mean()),
                   oracle=float(out[t]["oracle"]),
                   breakeven=float(out[t]["breakeven"]),
                   pearson=out[t]["pearson"], spearman=out[t]["spearman"])
           for t, _ in MODELS}, open(f"{BASE}\\stage_abcd.json", "w"), indent=1)
open(f"{BASE}\\report_abcd.txt", "w", encoding="utf-8").write("\n".join(R))
