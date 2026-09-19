"""Build bench/ref/overlap.html: for every annotated yes question, the temporal overlap of the served
27B configuration and of two extractive QA models (the "BERT" side) with the gold passage, on one
local time axis, plus the numbers that say whether one system can cover for the other.

Inputs, all in the repo:
  bench/results/probe/extractors/incumbent.json
      the 27B per question (span, gold, tIoU, binary): a replay of
      bench/results/llm/qwen3.8-27b.units-fewshot-both.large-v3-turbo.clause-and.p4.json, which is
      the served configuration (Qwen3.8-27B, units-fewshot-both, clause-and units, turbo transcripts)
  bench/results/probe/extractors/spans_{deberta-large,roberta-base}.json
      deepset/deberta-v3-large-squad2 and deepset/roberta-base-squad2, ZERO-SHOT, over the same turbo
      transcripts (sliding window 384/128, characters mapped to seconds by the served offsets with a
      word-boundary guard; run_zeroshot.py and spanlib.py next to them). Nikolaj's fine-tune on the
      medicalBertFinetune branch was never trained, so these are floors, not what a fine-tune would reach.
  bench/results/probe/extractors/tiou_*.json      stored per-question tIoU, used only as a cross-check
  data/question_train.csv, transcripts/conversation_*.large-v3-turbo.json

    python bench/overlap_page.py        # writes bench/ref/overlap.html and prints the summary

Open bench/ref/overlap.html directly, or http://localhost:9060/overlap.html under the oracle server.
Findings-log entry 61 holds the first analysis of the same files; this page is the visual of it.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
EXT = HERE / 'results' / 'probe' / 'extractors'
OUT = HERE / 'ref' / 'overlap.html'
ACC = 389 / 390          # the 27B's binary accuracy on the 390 training questions; unchanged by any span rule
SERIES = [
    ('q27', 'Qwen3.8-27B (served)', 'the served configuration: units-fewshot-both, clause-and units'),
    ('deb', 'DeBERTa-v3-large SQuAD2', 'deepset/deberta-v3-large-squad2, zero-shot'),
    ('rob', 'RoBERTa-base SQuAD2', 'deepset/roberta-base-squad2, zero-shot'),
]
FILES = {'deb': 'deberta-large', 'rob': 'roberta-base'}


def tiou(a, b):
    if not a or not b:
        return 0.0
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def score(m):
    return 0.4 * ACC + 0.6 * m


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


_words = {}


def words_of(tid):
    if tid not in _words:
        d = json.loads((CASE / 'transcripts' / f'conversation_{tid}.large-v3-turbo.json').read_text(encoding='utf-8'))
        _words[tid] = ([{'s': w['start'], 'e': w['end'], 'w': w['w'].strip()} for s in d['segments'] for w in (s.get('words') or [])],
                       float(d['duration']))
    return _words[tid]


def text_in(words, span):
    """Turbo words overlapping the interval by at least a third of their own duration (bench/timeline.py)."""
    if not span:
        return ''
    a, b = span
    out = []
    for w in words:
        d = max(1e-3, w['e'] - w['s'])
        if min(b, w['e']) - max(a, w['s']) >= d / 3:
            out.append(w['w'])
    return ' '.join(out)


def load():
    inc = json.loads((EXT / 'incumbent.json').read_text(encoding='utf-8'))
    qrow = {r['question_id']: r for r in csv.DictReader((CASE / 'data' / 'question_train.csv').open(encoding='utf-8'))}
    spans = {k: json.loads((EXT / f'spans_{t}.json').read_text(encoding='utf-8')) for k, t in FILES.items()}
    stored = {k: json.loads((EXT / f'tiou_{t}.json').read_text(encoding='utf-8')) for k, t in FILES.items()}
    rows = []
    for qid, v in inc.items():
        if v['gold'] is None:
            continue
        g = [float(v['gold'][0]), float(v['gold'][1])]
        assert abs(g[0] - float(qrow[qid]['evidence_start'])) < 1e-6 and abs(g[1] - float(qrow[qid]['evidence_end'])) < 1e-6, qid
        words, dur = words_of(v['transcript_id'])
        r = {'id': qid, 'tid': v['transcript_id'], 'n': int(v['transcript_id'].split('_')[1]), 'q': qrow[qid]['question'],
             'gold': g, 'dur': round(dur, 2), 'span': {}, 'tiou': {}, 'text': {'gold': text_in(words, g)},
             'binary': int(v['prediction'] == v['label'])}
        r['span']['q27'] = v['span']
        r['tiou']['q27'] = tiou(g, v['span'])
        assert abs(r['tiou']['q27'] - v['tiou']) < 1e-9, qid
        r['text']['q27'] = text_in(words, v['span'])
        for k in FILES:
            s = spans[k][qid]
            sp = [float(s['time_start']), float(s['time_end'])]
            r['span'][k] = sp
            r['tiou'][k] = tiou(g, sp)
            assert abs(r['tiou'][k] - stored[k][qid]) < 1e-6, (qid, k, r['tiou'][k], stored[k][qid])
            r['text'][k] = text_in(words, sp)
        rows.append(r)
    rows.sort(key=lambda r: (r['n'], r['id']))
    assert len(rows) == 195, len(rows)
    return rows


def analyse(rows):
    """Everything the page states in numbers, recomputed here from the spans."""
    q27 = [r['tiou']['q27'] for r in rows]
    out = {'n': len(rows), 'acc': ACC, 'q27': {'mean': mean(q27), 'zero': sum(t == 0 for t in q27), 'lt50': sum(t < .5 for t in q27),
                                                'ge60': sum(t >= .6 for t in q27)}, 'ext': {}}
    for k in FILES:
        e = [r['tiou'][k] for r in rows]
        gold = [r['gold'] for r in rows]
        s27 = [r['span']['q27'] for r in rows]
        se = [r['span'][k] for r in rows]
        agree = [tiou(a, b) for a, b in zip(s27, se)]
        union = [tiou(g, [min(a[0], b[0]), max(a[1], b[1])]) if a and b else t for g, a, b, t in zip(gold, s27, se, q27)]
        inter = [tiou(g, [max(a[0], b[0]), min(a[1], b[1])]) if a and b and min(a[1], b[1]) > max(a[0], b[0]) else t
                 for g, a, b, t in zip(gold, s27, se, q27)]
        route_dis = [x if ag > 0 else y for x, y, ag in zip(q27, e, agree)]        # disagree: trust the extractor
        lengths = [a[1] - a[0] if a else 0 for a in s27]
        best_len, best_L = mean(q27), None
        for L in sorted(set(lengths)):
            m = mean(y if l > L else x for x, y, l in zip(q27, e, lengths))
            if m > best_len + 1e-12:
                best_len, best_L = m, L
        zeros = [i for i, t in enumerate(q27) if t == 0]
        both_bad = [i for i, (x, y) in enumerate(zip(q27, e)) if x < .5 and y < .5]
        out['ext'][k] = {
            'mean': mean(e), 'zero': sum(t == 0 for t in e), 'lt50': sum(t < .5 for t in e), 'ge60': sum(t >= .6 for t in e),
            'rho': spearman(q27, e),
            'oracle': mean(max(x, y) for x, y in zip(q27, e)),
            'union': mean(union), 'inter': mean(inter), 'route_disagree': mean(route_dis),
            'route_len': best_len, 'route_len_L': best_L,
            'rescue': sum(x < .5 <= y for x, y in zip(q27, e)),
            'reverse': sum(y < .5 <= x for x, y in zip(q27, e)),
            'ahead25': sum(y - x >= .25 for x, y in zip(q27, e)),
            'behind25': sum(x - y >= .25 for x, y in zip(q27, e)),
            'zeros_ext_zero': sum(e[i] == 0 for i in zeros),
            'zeros_same_place': sum(agree[i] > .5 for i in zeros),
            'both_bad': len(both_bad), 'both_bad_same_place': sum(agree[i] > .5 for i in both_bad),
            'agree_mean': mean(agree), 'agree_ge90': sum(a >= .9 for a in agree),
        }
    q = [r['tiou']['q27'] for r in rows]
    out['oracle3'] = mean(max(x, r['tiou']['deb'], r['tiou']['rob']) for x, r in zip(q, rows))
    return out


def fmt(x, d=3):
    return f'{x:.{d}f}'


def summary_text(a):
    lines = [f"27B alone: mean tIoU {fmt(a['q27']['mean'], 4)}, score {fmt(score(a['q27']['mean']), 4)} at accuracy {fmt(ACC, 4)};"
             f" {a['q27']['zero']} of {a['n']} at exactly 0, {a['q27']['lt50']} below 0.5, {a['q27']['ge60']} at 0.6 or more"]
    for k, name, _ in SERIES[1:]:
        e = a['ext'][k]
        lines += [f"{name}: mean tIoU {fmt(e['mean'], 4)}, {e['zero']} at 0, {e['lt50']} below 0.5; Spearman rho with the 27B {fmt(e['rho'])}",
                  f"  pick the better of the two per question (needs a perfect router): {fmt(e['oracle'], 4)} tIoU, score {fmt(score(e['oracle']), 4)}"
                  f" ({score(e['oracle']) - score(a['q27']['mean']):+.4f})",
                  f"  union of both spans {fmt(e['union'], 4)}; intersection when they overlap else 27B {fmt(e['inter'], 4)};"
                  f" extractor when the two disagree {fmt(e['route_disagree'], 4)}; route on 27B span length (best in-sample cut) {fmt(e['route_len'], 4)}",
                  f"  rescues (27B < 0.5, extractor >= 0.5): {e['rescue']}; the reverse: {e['reverse']};"
                  f" extractor ahead by 0.25 or more: {e['ahead25']}, behind by 0.25 or more: {e['behind25']}",
                  f"  on the 27B's {a['q27']['zero']} whole misses the extractor is also 0 on {e['zeros_ext_zero']} and points at the same wrong passage on {e['zeros_same_place']};"
                  f" both below 0.5 on {e['both_bad']} questions, same wrong passage on {e['both_bad_same_place']}"]
    lines.append(f"best of all three per question: {fmt(a['oracle3'], 4)} tIoU, score {fmt(score(a['oracle3']), 4)}")
    return '\n'.join(lines)


TEMPLATE = r'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Temporal overlap</title>
<style>
  :root {
    color-scheme: light;
    --surface-1:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
    --ring:rgba(11,11,11,0.10); --gold:rgba(11,11,11,0.09); --gold-edge:#898781; --hi:rgba(42,120,214,0.10);
    --q27:#2a78d6; --deb:#eb6834; --rob:#1baf7a; --other:#b9b8b1;
  }
  @media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
    --ring:rgba(255,255,255,0.10); --gold:rgba(255,255,255,0.12); --gold-edge:#898781; --hi:rgba(57,135,229,0.16);
    --q27:#3987e5; --deb:#d95926; --rob:#199e70; --other:#4d4d4a;
  } }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
    --ring:rgba(255,255,255,0.10); --gold:rgba(255,255,255,0.12); --gold-edge:#898781; --hi:rgba(57,135,229,0.16);
    --q27:#3987e5; --deb:#d95926; --rob:#199e70; --other:#4d4d4a;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1280px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
  h2 { font-size:16px; font-weight:600; margin:32px 0 8px; }
  p { margin:6px 0; color:var(--ink-2); max-width:80ch; }
  p strong { color:var(--ink); font-weight:600; }
  .card { background:var(--surface-1); border:1px solid var(--ring); border-radius:8px; padding:16px; }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:16px 0; }
  .kpi .label { color:var(--ink-2); font-size:12px; }
  .kpi .value { font-size:28px; font-weight:600; line-height:1.2; margin-top:2px; }
  .kpi .note { color:var(--muted); font-size:12px; margin-top:2px; }
  .kpi .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:0; }
  .filters { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; margin:12px 0 16px; }
  .filters label { color:var(--ink-2); font-size:13px; display:inline-flex; gap:6px; align-items:center; }
  select { font:inherit; color:var(--ink); background:var(--surface-1); border:1px solid var(--axis); border-radius:6px; padding:4px 8px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 18px; color:var(--ink-2); font-size:13px; margin:4px 0 10px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .key { width:16px; height:8px; border-radius:4px; display:inline-block; }
  .key.gold { background:var(--gold); border-left:1px solid var(--gold-edge); border-right:1px solid var(--gold-edge); border-radius:0; height:14px; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; box-shadow:0 0 0 2px var(--surface-1); }
  .two { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; align-items:start; }
  @media (max-width:900px) { .two { grid-template-columns:1fr; } }
  svg { width:100%; height:auto; display:block; }
  svg text { font:12px system-ui,-apple-system,"Segoe UI",sans-serif; fill:var(--muted); font-variant-numeric:tabular-nums; }
  svg .title { fill:var(--ink-2); }
  table.stats { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  table.stats th, table.stats td { text-align:right; padding:5px 8px; border-bottom:1px solid var(--grid); white-space:nowrap; }
  table.stats th:first-child, table.stats td:first-child { text-align:left; white-space:normal; color:var(--ink-2); }
  table.stats th { color:var(--muted); font-weight:500; font-size:12px; }
  table.stats td.best { font-weight:600; }
  .rows { margin-top:8px; }
  .conv { color:var(--muted); font-size:12px; margin:14px 0 4px; padding-top:8px; border-top:1px solid var(--grid); font-variant-numeric:tabular-nums; }
  .row { display:grid; grid-template-columns:minmax(180px,300px) minmax(0,1fr) 168px; gap:12px; align-items:center; padding:5px 6px; border-radius:6px; }
  .row.flash { background:var(--hi); }
  .row .who { font-size:12px; min-width:0; }
  .row .who .id { color:var(--muted); font-variant-numeric:tabular-nums; }
  .row .who .q { color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .track { position:relative; height:60px; background:var(--surface-1); border-radius:4px; }
  .track .gold { position:absolute; top:0; bottom:0; background:var(--gold); border-left:1px solid var(--gold-edge); border-right:1px solid var(--gold-edge); }
  .track .bar { position:absolute; height:8px; border-radius:4px; }
  .track .bar::after { content:""; position:absolute; left:-4px; right:-4px; top:-7px; bottom:-7px; }   /* hit area larger than the mark */
  .track .bar:hover, .track .bar:focus { outline:2px solid var(--surface-1); box-shadow:0 0 0 3px currentColor; }
  .track .off { position:absolute; width:0; height:0; border-top:6px solid transparent; border-bottom:6px solid transparent; }
  .track .off.l { border-right:8px solid currentColor; left:2px; }
  .track .off.r { border-left:8px solid currentColor; right:2px; }
  .track .tick { position:absolute; bottom:2px; font-size:10px; color:var(--muted); font-variant-numeric:tabular-nums; pointer-events:none; }
  .track .tick.l { left:4px; } .track .tick.r { right:4px; }
  .nums { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; font-variant-numeric:tabular-nums; font-size:12px; text-align:right; }
  .nums span { display:inline-flex; align-items:center; justify-content:flex-end; gap:5px; }
  .nums .k { width:8px; height:8px; border-radius:2px; display:inline-block; }
  .nums .z { color:var(--muted); }
  #tip { position:fixed; z-index:9; pointer-events:none; background:var(--surface-1); color:var(--ink); border:1px solid var(--ring); box-shadow:0 4px 16px rgba(0,0,0,0.18); border-radius:6px; padding:8px 10px; font-size:12px; max-width:420px; display:none; }
  #tip .v { font-weight:600; font-size:13px; }
  #tip .s { color:var(--ink-2); }
  #tip .t { color:var(--ink-2); margin-top:4px; font-style:italic; }
  #tip .line { display:inline-block; width:12px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }
  .foot { color:var(--muted); font-size:12px; margin-top:24px; }
  .count { color:var(--muted); font-size:12px; margin-left:auto; }
</style>
<main>
  <h1>Temporal overlap: the served 27B against two extractive QA models</h1>
  <p>Every one of the __N__ annotated yes questions of the training set. The grey band is the annotated passage; each
     bar is what a system returned for that question; the number is its temporal intersection over union (tIoU), the
     part of the score worth 0.6. The 27B is the served configuration replayed offline. The two extractors are
     off-the-shelf SQuAD2 models run zero-shot on the same turbo transcripts, the way Nikolaj's branch would run them,
     but his fine-tune was never trained, so their level is a floor and not a fine-tune estimate.</p>

  <div class="kpis">__KPIS__</div>

  <div class="filters">
    <label>Conversation <select id="f-conv"><option value="all">all 39</option>__CONVS__</select></label>
    <label>Show <select id="f-show">
      <option value="all">all questions</option>
      <option value="q27lt50">27B below 0.5</option>
      <option value="q27zero">27B at 0</option>
      <option value="ahead">extractor ahead by 0.25 or more</option>
      <option value="rescue">extractor rescues (27B below 0.5, extractor 0.5 or more)</option>
      <option value="bothbad">both below 0.5</option>
    </select></label>
    <label>Sort <select id="f-sort">
      <option value="conv">by conversation</option>
      <option value="q27">worst 27B first</option>
      <option value="gain">largest extractor gain first</option>
    </select></label>
    <label>Extractor on the scatter <select id="f-ext"><option value="deb">DeBERTa-v3-large</option><option value="rob">RoBERTa-base</option></select></label>
    <span class="count" id="count"></span>
  </div>

  <div class="two">
    <div class="card">
      <div class="legend"><span><i class="dot" style="background:var(--deb)"></i>extractor ahead by 0.25 or more</span><span><i class="dot" style="background:var(--q27)"></i>every other question</span></div>
      <div id="scatter"></div>
      <p style="font-size:12px;color:var(--muted)">One point per question. Above the diagonal the extractor found the passage better than the 27B did. Hover the nearest point; click it to jump to its row.</p>
    </div>
    <div class="card">
      <table class="stats" id="stats">__STATS__</table>
      <p style="font-size:12px;color:var(--muted);margin-top:8px">Score = 0.4 x accuracy + 0.6 x mean tIoU, with the 27B's accuracy (__ACC__) held fixed: routing spans never changes the binary half. "Pick the better" assumes a router that is always right; no signal measured so far (entry 61) routes better than always choosing the 27B. The length cut is fitted on these same questions and is optimistic.</p>
    </div>
  </div>

  <h2>Per question, on a local time axis</h2>
  <div class="legend">
    <span><i class="key gold"></i>annotated passage</span>
    <span><i class="key" style="background:var(--q27)"></i>Qwen3.8-27B (served)</span>
    <span><i class="key" style="background:var(--deb)"></i>DeBERTa-v3-large, zero-shot</span>
    <span><i class="key" style="background:var(--rob)"></i>RoBERTa-base, zero-shot</span>
    <span style="color:var(--muted)">a triangle at the edge: that span lies outside the window shown; hover it for the time</span>
  </div>
  <div class="card rows" id="rows"></div>

  <p class="foot">Generated by bench/overlap_page.py from bench/results/probe/extractors (the files behind findings-log entry 61),
     data/question_train.csv and the turbo transcripts. Words shown in the tooltips are the transcript words under each span.</p>
</main>
<div id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const rows = DATA.rows;
  const NAMES = DATA.names;                      // key -> display name
  const KEYS = ['q27', 'deb', 'rob'];
  const state = { conv: 'all', show: 'all', sort: 'conv', ext: 'deb' };
  const tip = document.getElementById('tip');
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const f3 = (x) => x.toFixed(3);
  const f2 = (x) => x.toFixed(2);

  function filtered() {
    const e = state.ext;
    let list = rows.filter(r => state.conv === 'all' || r.tid === state.conv);
    list = list.filter(r => {
      const a = r.tiou.q27, b = r.tiou[e];
      switch (state.show) {
        case 'q27lt50': return a < .5;
        case 'q27zero': return a === 0;
        case 'ahead': return b - a >= .25;
        case 'rescue': return a < .5 && b >= .5;
        case 'bothbad': return a < .5 && b < .5;
        default: return true;
      }
    });
    if (state.sort === 'q27') list.sort((x, y) => x.tiou.q27 - y.tiou.q27 || x.n - y.n);
    else if (state.sort === 'gain') list.sort((x, y) => (y.tiou[e] - y.tiou.q27) - (x.tiou[e] - x.tiou.q27) || x.n - y.n);
    else list.sort((x, y) => x.n - y.n || (x.id < y.id ? -1 : 1));
    return list;
  }

  // ---- tooltip -------------------------------------------------------------------------------------
  function showTip(ev, parts) {
    tip.replaceChildren();
    for (const p of parts) {
      const d = document.createElement('div'); d.className = p.cls || '';
      if (p.color) { const l = document.createElement('i'); l.className = 'line'; l.style.background = p.color; d.appendChild(l); }
      d.appendChild(document.createTextNode(p.text)); tip.appendChild(d);
    }
    tip.style.display = 'block'; moveTip(ev);
  }
  function moveTip(ev) {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { tip.style.display = 'none'; }

  function rowParts(r) {
    const e = state.ext;
    const parts = [{ text: r.id + '  ' + r.q, cls: 'v' }];
    for (const k of KEYS) parts.push({ text: `${f3(r.tiou[k])}  ${NAMES[k]}  ${f2(r.span[k][0])} to ${f2(r.span[k][1])} s`, color: css('--' + k), cls: 's' });
    parts.push({ text: `gold ${f2(r.gold[0])} to ${f2(r.gold[1])} s: "${r.text.gold}"`, cls: 't' });
    return parts;
  }

  // ---- scatter -------------------------------------------------------------------------------------
  const NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, parent) {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function renderScatter(list) {
    const host = document.getElementById('scatter'); host.replaceChildren();
    const W = 520, H = 440, L = 44, R = 14, T = 14, B = 44, pw = W - L - R, ph = H - T - B;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'tIoU of the 27B against the extractor, one point per question' }, host);
    const X = v => L + v * pw, Y = v => T + ph - v * ph;
    for (const t of [0, .25, .5, .75, 1]) {
      el('line', { x1: X(t), x2: X(t), y1: T, y2: T + ph, stroke: css('--grid'), 'stroke-width': 1 }, svg);
      el('line', { y1: Y(t), y2: Y(t), x1: L, x2: L + pw, stroke: css('--grid'), 'stroke-width': 1 }, svg);
      const tx = el('text', { x: X(t), y: T + ph + 16, 'text-anchor': 'middle' }, svg); tx.textContent = t.toFixed(2);
      const ty = el('text', { x: L - 6, y: Y(t) + 4, 'text-anchor': 'end' }, svg); ty.textContent = t.toFixed(2);
    }
    el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: css('--axis'), 'stroke-width': 1 }, svg);
    const xl = el('text', { x: L + pw / 2, y: H - 6, 'text-anchor': 'middle', class: 'title' }, svg); xl.textContent = 'tIoU, ' + NAMES.q27;
    const yl = el('text', { x: 12, y: T + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${T + ph / 2})`, class: 'title' }, svg); yl.textContent = 'tIoU, ' + NAMES[state.ext];
    const pts = [];
    const g = el('g', {}, svg);
    for (const r of list) {
      const a = r.tiou.q27, b = r.tiou[state.ext];
      const ahead = b - a >= .25;
      const c = el('circle', { cx: X(a), cy: Y(b), r: 4.5, fill: css(ahead ? '--deb' : '--q27'), stroke: css('--surface-1'), 'stroke-width': 2 }, g);
      pts.push({ x: X(a), y: Y(b), r, c });
    }
    // nearest-point hover over the whole plot
    const hit = el('rect', { x: L, y: T, width: pw, height: ph, fill: 'transparent', style: 'cursor:crosshair' }, svg);
    let cur = null;
    const pt = svg.createSVGPoint();
    function nearest(ev) {
      pt.x = ev.clientX; pt.y = ev.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      let best = null, bd = 1e9;
      for (const q of pts) { const d = Math.hypot(q.x - p.x, q.y - p.y); if (d < bd) { bd = d; best = q; } }
      return bd < 40 ? best : null;
    }
    hit.addEventListener('pointermove', ev => {
      const q = nearest(ev);
      if (cur && cur !== q) cur.c.setAttribute('r', 4.5);
      cur = q;
      if (!q) { hideTip(); return; }
      q.c.setAttribute('r', 7); g.appendChild(q.c);
      showTip(ev, rowParts(q.r));
    });
    hit.addEventListener('pointerleave', () => { if (cur) cur.c.setAttribute('r', 4.5); cur = null; hideTip(); });
    hit.addEventListener('click', ev => {
      const q = nearest(ev); if (!q) return;
      const node = document.getElementById('row-' + q.r.id);
      if (!node) return;
      node.scrollIntoView({ behavior: 'smooth', block: 'center' });
      node.classList.add('flash'); setTimeout(() => node.classList.remove('flash'), 2500);
    });
  }

  // ---- per-question timelines ----------------------------------------------------------------------
  function windowFor(r) {
    const PAD = 3, MIN = 16, CAP = 45;
    let lo = Math.max(0, Math.min(r.gold[0], r.span.q27[0]) - PAD);
    let hi = Math.min(r.dur, Math.max(r.gold[1], r.span.q27[1]) + PAD);
    if (hi - lo < MIN) { const c = (lo + hi) / 2; lo = Math.max(0, c - MIN / 2); hi = Math.min(r.dur, lo + MIN); lo = Math.max(0, hi - MIN); }
    for (const k of ['deb', 'rob']) {
      const s = r.span[k];
      const nlo = Math.min(lo, Math.max(0, s[0] - PAD)), nhi = Math.max(hi, Math.min(r.dur, s[1] + PAD));
      if (nhi - nlo <= CAP) { lo = nlo; hi = nhi; }
    }
    return [lo, hi];
  }
  function renderRows(list) {
    const host = document.getElementById('rows'); host.replaceChildren();
    document.getElementById('count').textContent = list.length + ' of ' + rows.length + ' questions';
    let lastConv = null;
    const LANES = { q27: 5, deb: 19, rob: 33 };
    for (const r of list) {
      if (state.sort === 'conv' && r.tid !== lastConv) {
        lastConv = r.tid;
        const h = document.createElement('div'); h.className = 'conv';
        h.textContent = `${r.tid}   ${f2(r.dur)} s   ${rows.filter(x => x.tid === r.tid).length} yes questions`;
        host.appendChild(h);
      }
      const row = document.createElement('div'); row.className = 'row'; row.id = 'row-' + r.id;
      const who = document.createElement('div'); who.className = 'who';
      const id = document.createElement('div'); id.className = 'id'; id.textContent = r.id; who.appendChild(id);
      const q = document.createElement('div'); q.className = 'q'; q.textContent = r.q; q.title = r.q; who.appendChild(q);
      row.appendChild(who);
      const track = document.createElement('div'); track.className = 'track';
      const [lo, hi] = windowFor(r);
      const px = t => ((t - lo) / (hi - lo) * 100);
      const gold = document.createElement('div'); gold.className = 'gold';
      gold.style.left = px(r.gold[0]) + '%'; gold.style.width = Math.max(0.3, px(r.gold[1]) - px(r.gold[0])) + '%';
      track.appendChild(gold);
      for (const k of KEYS) {
        const s = r.span[k], color = css('--' + k);
        const parts = [{ text: `${f3(r.tiou[k])}  tIoU`, cls: 'v' },
                       { text: `${NAMES[k]}  ${f2(s[0])} to ${f2(s[1])} s`, color, cls: 's' },
                       { text: `"${r.text[k]}"`, cls: 't' },
                       { text: `gold ${f2(r.gold[0])} to ${f2(r.gold[1])} s: "${r.text.gold}"`, cls: 't' }];
        let node;
        if (s[1] < lo || s[0] > hi) {
          node = document.createElement('div'); node.className = 'off ' + (s[1] < lo ? 'l' : 'r'); node.style.color = color;
          node.style.top = (LANES[k] - 2) + 'px';
        } else {
          node = document.createElement('div'); node.className = 'bar'; node.style.background = color; node.style.color = color;
          node.style.top = LANES[k] + 'px';
          const a = Math.max(lo, s[0]), b = Math.min(hi, s[1]);
          node.style.left = px(a) + '%'; node.style.width = Math.max(0.4, px(b) - px(a)) + '%';
        }
        node.tabIndex = 0;
        node.addEventListener('pointerenter', ev => showTip(ev, parts));
        node.addEventListener('pointermove', moveTip);
        node.addEventListener('pointerleave', hideTip);
        node.addEventListener('focus', ev => { const b = node.getBoundingClientRect(); showTip({ clientX: b.left, clientY: b.bottom }, parts); });
        node.addEventListener('blur', hideTip);
        track.appendChild(node);
      }
      const tl = document.createElement('span'); tl.className = 'tick l'; tl.textContent = f2(lo) + ' s'; track.appendChild(tl);
      const tr = document.createElement('span'); tr.className = 'tick r'; tr.textContent = f2(hi) + ' s'; track.appendChild(tr);
      row.appendChild(track);
      const nums = document.createElement('div'); nums.className = 'nums';
      for (const k of KEYS) {
        const sp = document.createElement('span'); if (r.tiou[k] === 0) sp.className = 'z';
        const key = document.createElement('i'); key.className = 'k'; key.style.background = css('--' + k); sp.appendChild(key);
        sp.appendChild(document.createTextNode(f3(r.tiou[k]))); nums.appendChild(sp);
      }
      row.appendChild(nums);
      host.appendChild(row);
    }
  }

  function render() { const list = filtered(); renderScatter(list); renderRows(list); }
  for (const [id, key] of [['f-conv', 'conv'], ['f-show', 'show'], ['f-sort', 'sort'], ['f-ext', 'ext']]) {
    document.getElementById(id).addEventListener('change', ev => { state[key] = ev.target.value; render(); });
  }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
  render();
})();
</script>
</html>
'''


def kpi(label, value, note='', color=None):
    sw = f'<i class="swatch" style="background:var(--{color})"></i>' if color else ''
    return f'<div class="card kpi"><div class="label">{sw}{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'


def build(rows, a):
    d = a['ext']['deb']
    kpis = [
        kpi('27B, mean tIoU', fmt(a['q27']['mean']), f"score {fmt(score(a['q27']['mean']))} at accuracy {fmt(ACC)}", 'q27'),
        kpi('DeBERTa-v3-large, zero-shot', fmt(d['mean']), f"{d['zero']} of {a['n']} at 0", 'deb'),
        kpi('RoBERTa-base, zero-shot', fmt(a['ext']['rob']['mean']), f"{a['ext']['rob']['zero']} of {a['n']} at 0", 'rob'),
        kpi('Pick the better of 27B and DeBERTa', fmt(d['oracle']), f"score {fmt(score(d['oracle']))}, {score(d['oracle']) - score(a['q27']['mean']):+.3f}; needs a router that is always right"),
        kpi('Questions DeBERTa rescues', str(d['rescue']), f"27B below 0.5 and DeBERTa at 0.5 or more; the reverse: {d['reverse']}"),
        kpi('27B whole misses also missed', f"{d['zeros_ext_zero']} of {a['q27']['zero']}", f"DeBERTa points at the same wrong passage on {d['zeros_same_place']} of them"),
    ]
    convs = ''.join(f'<option value="{t}">{t}</option>' for t in sorted({r['tid'] for r in rows}, key=lambda t: int(t.split('_')[1])))
    m27 = a['q27']['mean']

    def line(name, vals, best=None):
        cells = ''.join(f'<td class="{"best" if best is not None and abs(v - best) < 1e-12 else ""}">{fmt(v)}</td>' for v in vals)
        return f'<tr><td>{name}</td>{cells}</tr>'
    stats = ['<tr><th>span rule, mean tIoU over the 195 gold questions</th><th>with DeBERTa</th><th>with RoBERTa</th><th>score, DeBERTa</th></tr>']
    e1, e2 = a['ext']['deb'], a['ext']['rob']
    stats.append(line('27B alone (served)', [m27, m27, score(m27)]))
    stats.append(line('extractor alone', [e1['mean'], e2['mean'], score(e1['mean'])]))
    stats.append(line('pick the better per question (perfect router)', [e1['oracle'], e2['oracle'], score(e1['oracle'])], max(e1['oracle'], e2['oracle'])))
    stats.append(line('union of the two spans', [e1['union'], e2['union'], score(e1['union'])]))
    stats.append(line('intersection when they overlap, else 27B', [e1['inter'], e2['inter'], score(e1['inter'])]))
    stats.append(line('extractor when the two spans do not touch, else 27B', [e1['route_disagree'], e2['route_disagree'], score(e1['route_disagree'])]))
    stats.append(line('extractor when the 27B span is long (best in-sample cut)', [e1['route_len'], e2['route_len'], score(e1['route_len'])]))
    stats.append(line('best of all three per question', [a['oracle3'], a['oracle3'], score(a['oracle3'])]))
    stats.append(f'<tr><td>Spearman rho with the 27B, per question</td><td>{fmt(e1["rho"])}</td><td>{fmt(e2["rho"])}</td><td></td></tr>')
    stats.append(f'<tr><td>questions where both are below 0.5 / same wrong passage</td><td>{e1["both_bad"]} / {e1["both_bad_same_place"]}</td><td>{e2["both_bad"]} / {e2["both_bad_same_place"]}</td><td></td></tr>')
    data = {'rows': rows, 'names': {k: n for k, n, _ in SERIES}}
    html = (TEMPLATE.replace('__N__', str(a['n'])).replace('__KPIS__', ''.join(kpis)).replace('__CONVS__', convs)
            .replace('__STATS__', ''.join(stats)).replace('__ACC__', fmt(ACC))
            .replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')))
    return html


def main():
    rows = load()
    a = analyse(rows)
    OUT.write_text(build(rows, a), encoding='utf-8')
    print(summary_text(a))
    print(f'wrote {OUT} ({OUT.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
