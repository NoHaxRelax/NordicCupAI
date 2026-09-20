"""Counterfactual probe on the EVALUATION conversations (no labels exist for them).

The one evaluation attempt was served by the 27B on 2026-09-20 (score 0.7814). The pod's request dump
holds the 38 evaluation conversations (audio + questions) and what we answered. This script lets another
model answer the same questions offline, with every labelled conversation we own (39 training + 19
validation) as worked examples, and compares its answers with the served ones. Nothing here can be
scored: the evaluation labels are hidden. What CAN be read: the evaluation set is exactly balanced
(README), so the true number of yes questions is half of all questions; each model's yes count against
that, the binary disagreements (readable against the transcript), and how often the spans coincide.

    python bench/llm/probe_eval.py transcribe --dir request_dump/eval
    python bench/llm/probe_eval.py dump --dir request_dump/eval --out bench/results/probe/prompts-eval-all
    ... one tool-less agent per conversation writes <answers>/<stem>.json ...
    python bench/llm/probe_eval.py compare --dir request_dump/eval --answers bench/results/probe/sonnet-eval \
        --served bench/results/served/pod4/answers.jsonl --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import os
import site
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent.parent
for p in (str(HERE), str(CASE), str(CASE / 'bench' / 'mine')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('ASR_MODEL', 'large-v3-turbo')
os.environ.setdefault('UNIT_SPLIT', 'clause-and')
ASR = 'large-v3-turbo'


def stems(d: Path):
    return sorted((p.name.split('.questions.json')[0] for p in d.glob('*.questions.json')), key=lambda s: int(s.split('_')[-1]))


def transcribe(a):
    _nv = Path(site.getsitepackages()[0]) / 'Lib' / 'site-packages' / 'nvidia'
    for sub in ('cublas', 'cudnn', 'cuda_nvrtc'):
        p = _nv / sub / 'bin'
        if p.is_dir():
            os.add_dll_directory(str(p)); os.environ['PATH'] = str(p) + os.pathsep + os.environ['PATH']
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    from faster_whisper import WhisperModel
    d = Path(a.dir); out = d / 'transcripts'; out.mkdir(exist_ok=True)
    model = WhisperModel(ASR, device='cuda', compute_type='float16')
    kw = dict(language='en', word_timestamps=True, beam_size=5, vad_filter=False)      # model.asr_kwargs() as served (ASR_CLEAN=0)
    for s in stems(d):
        f = out / f'{s}.{ASR}.json'
        if f.exists():
            continue
        t0 = time.time()
        segments, info = model.transcribe(str(d / f'{s}.mp3'), **kw)
        segs = [{'start': x.start, 'end': x.end, 'text': x.text,
                 'words': [{'w': w.word, 'start': w.start, 'end': w.end, 'p': w.probability} for w in (x.words or [])]} for x in segments]
        f.write_text(json.dumps({'file': f'{s}.mp3', 'model': ASR, 'duration': info.duration, 'seconds': time.time() - t0,
                                 'word_timestamps': True, 'segments': segs}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'{s}: {info.duration:.0f} s audio in {time.time()-t0:.1f} s', flush=True)


def conversations(d: Path):
    import model
    for s in stems(d):
        t = json.loads((d / 'transcripts' / f'{s}.{ASR}.json').read_text(encoding='utf-8'))
        words = []
        for seg in t['segments']:
            for w in seg.get('words', []):
                words.append(model.Word(w['w'], float(w['start']), float(w['end'])))
            if seg.get('words'):
                words[-1].w += '\x00'
        yield s, json.loads((d / f'{s}.questions.json').read_text(encoding='utf-8')), words, float(t['duration']), model.make_units(words)


def dump(a):
    from prompts import VARIANTS
    v = VARIANTS[a.variant]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s, qs, words, duration, units in conversations(Path(a.dir)):
        v.set_conversation(s, ASR)                      # not in the pool: every labelled conversation is a demo
        p = v.build_all([q.strip() for q in qs], units)
        txt = [f'### SYSTEM\n{p.system}']
        for du, da in (p.demos or []):
            txt.append(f'### EXAMPLE INPUT\n{du}\n### EXAMPLE OUTPUT\n{da}')
        txt.append(f'### INPUT\n{p.user}')
        text = '\n\n'.join(txt)
        (out / f'{s}.txt').write_text(text, encoding='utf-8')
        manifest.append({'stem': s, 'demos': len(p.demos or []), 'chars': len(text), 'lines': text.count('\n') + 1})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'{len(manifest)} prompts -> {out}; demos {manifest[0]["demos"]}; about {statistics.mean(m["chars"] for m in manifest)/4000:.0f}k tokens each')


def tiou(x, y):
    lo, hi = max(x[0], y[0]), min(x[1], y[1])
    inter = max(0.0, hi - lo); union = max(x[1], y[1]) - min(x[0], y[0])
    return inter / union if union > 0 else 0.0


def compare(a):
    from prompts import VARIANTS
    v = VARIANTS[a.variant]
    served = {}
    for line in Path(a.served).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line); served[Path(r['file']).stem] = r          # the last record per conversation wins
    n = yes_m = yes_s = same_bin = 0
    both_yes, agree_sp, pair_iou, dis = 0, 0, [], []
    for s, qs, words, duration, units in conversations(Path(a.dir)):
        f = Path(a.answers) / f'{s}.json'
        if not f.exists():
            print('missing', s); continue
        v.set_conversation(s, ASR)
        p = v.build_all([q.strip() for q in qs], units)
        per = v.split(json.loads(f.read_text(encoding='utf-8')), len(qs))
        sv = served[s]
        for i, (q, item) in enumerate(zip(qs, per)):
            yes, span = p.postprocess(item, units, words, duration)
            sy, ss = bool(sv['answers'][i]), sv['spans'][i]
            n += 1; yes_m += bool(yes); yes_s += sy; same_bin += (bool(yes) == sy)
            if bool(yes) != sy:
                dis.append({'conv': s, 'q': i + 1, 'question': q, 'served': sy, a.model: bool(yes), 'quote': item.get('quote', ''),
                            'segments': item.get('segments', []), 'served_span': ss, 'model_span': list(span) if span else None,
                            'span_tiou': tiou(span, ss) if (span and ss) else None})
            if yes and sy and span and ss:
                both_yes += 1; t = tiou(span, ss); pair_iou.append(t); agree_sp += (t > 0.999)
    print(f'{n} questions; balanced set, so true yes = {n//2}')
    print(f'  served 27B said yes on {yes_s}; {a.model} says yes on {yes_m}; same binary on {same_bin} of {n}')
    print(f'  both yes on {both_yes}: identical span on {agree_sp}, mean tIoU between the two spans {statistics.mean(pair_iou):.3f}, '
          f'disjoint on {sum(t == 0 for t in pair_iou)}')
    print(f'  binary disagreements: {len(dis)}  ({sum(1 for x in dis if x[a.model])} where only {a.model} says yes, '
          f'{sum(1 for x in dis if x["served"])} where only the served 27B says yes)')
    for x in dis:
        st = 'we sent no span' if not x['served_span'] else (f"our span vs its span tIoU {x['span_tiou']:.2f}" if x['span_tiou'] is not None else f"we sent {x['served_span']}")
        print(f"   {x['conv'].replace('conversation_', '')} q{x['q']}: served {'yes' if x['served'] else 'no'}, {a.model} {'yes' if x[a.model] else 'no'} | {st} | {x['question']} | quote: {x['quote'][:110]!r}")
    Path(a.answers, '_compare.json').write_text(json.dumps({'questions': n, 'served_yes': yes_s, 'model_yes': yes_m, 'same_binary': same_bin,
                                                            'both_yes': both_yes, 'identical_spans': agree_sp, 'disagreements': dis},
                                                           ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['transcribe', 'dump', 'compare'])
    ap.add_argument('--dir', default=str(CASE / 'request_dump' / 'eval'))
    ap.add_argument('--variant', default='units-joint-demo-all-both-val')
    ap.add_argument('--out', default=str(CASE / 'bench' / 'results' / 'probe' / 'prompts-eval-all'))
    ap.add_argument('--answers', default='')
    ap.add_argument('--served', default=str(CASE / 'bench' / 'results' / 'served' / 'pod4' / 'answers.jsonl'))
    ap.add_argument('--model', default='model')
    a = ap.parse_args()
    {'transcribe': transcribe, 'dump': dump, 'compare': compare}[a.cmd](a)
