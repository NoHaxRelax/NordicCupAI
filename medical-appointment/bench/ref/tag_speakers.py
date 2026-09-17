"""Pre-fill [D]/[P] speaker tags in bench/ref/*.txt so the human oracle only
flips the wrong ones.

Per conversation: each line's audio (from its [m:ss.s] marker to the next
line's marker) is summarised by its mean MFCC vector, the lines are split into
two clusters (two voices), and the cluster whose lines read like the clinician
(second-person address, instructions, findings) is labelled D. Lines that
already carry a tag are left alone; the text is never touched.

    python bench/ref/tag_speakers.py                 # all files
    python bench/ref/tag_speakers.py --only sample_5 # one file
    python bench/ref/tag_speakers.py --skip sample_5 # everything but one

Prints, per file, the number of lines tagged and a separation score (higher is
more confident; below ~1.0 means the voices were hard to tell apart, so check
that file more carefully).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # bench/ref
CASE = HERE.parent.parent                          # medical-appointment
sys.path.insert(0, str(CASE / 'bench' / 'asr'))
from common import load_audio  # noqa: E402

MARK = re.compile(r'^\[(\d+):(\d+(?:\.\d+)?)\]\s*')
TAG = re.compile(r'^\[(D|P|\?)\]\s*')

DOCTOR_CUES = re.compile(
    r"\b(let me|i'll prescribe|i will prescribe|i recommend|i'd like you|i would like you|"
    r"take (one|two|a|the)|twice a day|once a day|tablets?|prescription|prescribe|we'll (book|see|check)|"
    r"come back|follow[- ]up|your (chest|heart|blood|results?|lungs|throat|ears?|test|dose|medication|prescription)|"
    r"breathe|sounds? normal|nothing abnormal|examination|i can see|open your mouth|any (pain|fever|nausea))\b", re.I)
PATIENT_CUES = re.compile(
    r"\b(i feel|i've been|i have been|my (chest|stomach|head|back|knee|throat|ears?|doctor|wife|husband|son|daughter|"
    r"symptoms|medication|pills|tablets)|i've had|i have had|it hurts|i'm worried|i am worried|should i|"
    r"is (it|that) (serious|bad|normal)|thank you,? doctor|doctor,)\b", re.I)


def parse(path: Path):
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for i, line in enumerate(lines):
        m = MARK.match(line)
        if not m:
            rows.append((i, None, None, line)); continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        rest = line[m.end():]
        tag = TAG.match(rest)
        rows.append((i, t, tag.group(1) if tag else None, rest[tag.end():] if tag else rest))
    return lines, rows


def features(audio, sr, spans):
    import librosa
    feats = []
    for (a, b) in spans:
        seg = audio[int(a * sr): int(max(b, a + 0.3) * sr)]
        if len(seg) < sr // 4:
            seg = np.pad(seg, (0, sr // 4 - len(seg)))
        mf = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
        f0 = librosa.yin(seg, fmin=60, fmax=400, sr=sr)
        feats.append(np.concatenate([mf.mean(axis=1), mf.std(axis=1), [np.median(f0)]]))
    X = np.array(feats)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    return X


def two_means(X, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(8):
        c = X[rng.choice(len(X), 2, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            new = np.array([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
            if np.allclose(new, c):
                break
            c = new
        inertia = d.min(1).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), c.copy())
    _, lab, c = best
    # A cluster holding under 15% of the lines is an outlier (a cough, a very
    # short line, a decode glitch), not a voice. Re-cluster without it, then
    # assign the outliers to the nearest real centroid.
    small = np.bincount(lab, minlength=2).argmin()
    if (lab == small).sum() < max(2, int(0.15 * len(X))) and (lab != small).sum() >= 4:
        keep = lab != small
        lab2, sep2 = two_means(X[keep], iters, seed + 1)
        c2 = np.array([X[keep][lab2 == k].mean(0) for k in range(2)])
        full = np.zeros(len(X), dtype=int)
        full[keep] = lab2
        d = ((X[~keep][:, None, :] - c2[None, :, :]) ** 2).sum(-1)
        full[~keep] = d.argmin(1)
        return full, sep2
    # separation: distance between centroids over mean within-cluster spread
    within = np.mean([np.sqrt(((X[lab == k] - c[k]) ** 2).sum(1)).mean() for k in range(2) if (lab == k).any()])
    sep = float(np.sqrt(((c[0] - c[1]) ** 2).sum()) / (within + 1e-6))
    return lab, sep


def llm_roles(lines):
    """Ask the local Ollama model (LLM_MODEL, default qwen3:4b) who speaks each
    line. Returns an int array with 1 = doctor, or None on any failure."""
    import json
    import os
    import requests
    url = os.environ.get('LLM_URL', 'http://localhost:11434').rstrip('/')
    model = os.environ.get('LLM_MODEL', 'qwen3:4b')
    numbered = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(lines))
    schema = {'type': 'object',
              'properties': {'speakers': {'type': 'array', 'minItems': len(lines), 'maxItems': len(lines),
                                          'items': {'type': 'object',
                                                    'properties': {'line': {'type': 'integer'},
                                                                   'speaker': {'type': 'string', 'enum': ['D', 'P']}},
                                                    'required': ['line', 'speaker']}}},
              'required': ['speakers']}
    body = {'model': model, 'stream': False, 'think': False, 'format': schema,
            'options': {'temperature': 0.0, 'num_ctx': 6144, 'num_predict': 30 * len(lines) + 60},
            'keep_alive': -1,
            'messages': [
                {'role': 'system', 'content':
                 'A doctor and a patient are talking. The transcript is a numbered list of lines; '
                 'a line can be a continuation by the same speaker as the previous line. For every '
                 'line decide who says it: D for the doctor (asks about symptoms, examines, explains '
                 'results, prescribes, gives instructions, introduces themself as Dr) or P for the '
                 'patient (describes their own symptoms, history, worries, asks what to do). '
                 f'Return JSON {{"speakers": [{{"line": 1, "speaker": "D"}}, ...]}} with one entry per line, '
                 f'exactly {len(lines)} entries, line numbers 1 to {len(lines)} in order.'},
                {'role': 'user', 'content': numbered}]}
    try:
        r = requests.post(f'{url}/api/chat', json=body, timeout=180)
        r.raise_for_status()
        sp = json.loads(r.json()['message']['content'])['speakers']
        by_line = {}
        for e in sp:
            try:
                by_line.setdefault(int(e['line']), e['speaker'])
            except (KeyError, TypeError, ValueError):
                continue
        missing = [i + 1 for i in range(len(lines)) if (i + 1) not in by_line]
        if len(missing) > len(lines) // 4:
            print(f'   llm labelled {len(lines) - len(missing)} of {len(lines)} lines; ignored')
            return None
        if missing:
            print(f'   llm missed lines {missing}; filled by alternation from the previous line')
        out = []
        for i in range(len(lines)):
            s = by_line.get(i + 1)
            if s is None:
                s = 'P' if (out and out[-1] == 1) else 'D'
            out.append(1 if s == 'D' else 0)
        return np.array(out)
    except Exception as e:  # noqa: BLE001
        print(f'   llm failed: {e}')
        return None


def line_cue(text: str) -> float:
    """Positive = reads like the clinician, negative = like the patient."""
    s = len(DOCTOR_CUES.findall(text)) - len(PATIENT_CUES.findall(text))
    s += 0.3 * len(re.findall(r"\byour?\b", text, re.I)) - 0.3 * len(re.findall(r"\b(i|my|me)\b", text, re.I))
    return float(max(-3.0, min(3.0, s)))


def viterbi(X, lab, sep, cues, p_switch=0.85, gamma=0.6):
    """Decode twice, once with state 0 as the doctor and once with state 1,
    adding the wording cue as evidence for the doctor state; keep the higher
    scoring path and return labels where 1 = doctor."""
    best = None
    for doctor in (0, 1):
        path, total = _viterbi(X, lab, sep, cues, doctor, p_switch, gamma)
        if best is None or total > best[0]:
            best = (total, path, doctor)
    _, path, doctor = best
    return (path == doctor).astype(int)


def _viterbi(X, lab, sep, cues, doctor, p_switch, gamma):
    """Two-state sequence decode. Prior: speakers alternate with probability
    p_switch per line. Evidence: distance of each line to the two voice
    centroids from the clustering, scaled by the within-cluster spread, so on
    a file where the voices are alike the evidence is weak and alternation
    decides, and on a file with distinct voices the audio overrides the prior
    where two consecutive lines are the same voice."""
    c = np.array([X[lab == k].mean(0) if (lab == k).any() else X.mean(0) for k in range(2)])
    spread = np.mean([np.sqrt(((X[lab == k] - c[k]) ** 2).sum(1)).mean() for k in range(2) if (lab == k).any()]) + 1e-6
    d = np.sqrt(((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)) / spread      # n x 2, in units of spread
    # emission log-likelihood: the more separated the voices, the sharper
    weight = min(3.0, max(0.2, sep))
    emit = -0.5 * weight * d ** 2
    emit[:, doctor] += gamma * cues
    emit[:, 1 - doctor] -= gamma * cues
    trans = np.log(np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]]))
    n = len(X)
    score = np.zeros((n, 2)); back = np.zeros((n, 2), dtype=int)
    score[0] = emit[0]
    for i in range(1, n):
        for k in range(2):
            cand = score[i - 1] + trans[:, k]
            back[i, k] = cand.argmax(); score[i, k] = cand.max() + emit[i, k]
    path = np.zeros(n, dtype=int); path[-1] = score[-1].argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, float(score[-1].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='e.g. sample_5')
    ap.add_argument('--skip', help='e.g. sample_5')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--force', action='store_true', help='replace existing tags (only on files nobody has hand-tagged)')
    ap.add_argument('--alternate', action='store_true', help='ignore audio; assume strict turn alternation')
    ap.add_argument('--llm', action='store_true', help='always use the local LLM for roles (default: only when voices do not separate)')
    a = ap.parse_args()
    files = sorted(HERE.glob('conversation_sample_*.txt'), key=lambda p: int(re.sub(r'\D', '', p.stem)))
    for path in files:
        tid = path.stem.replace('conversation_', '')
        if a.only and tid != a.only: continue
        if a.skip and tid == a.skip: continue
        lines, rows = parse(path)
        if a.force:
            rows = [(i, t, None, text) for (i, t, tag, text) in rows]
        timed = [r for r in rows if r[1] is not None]
        if len(timed) < 3:
            print(f'{tid}: too few lines, skipped'); continue
        audio, sr = load_audio(CASE / 'data' / 'audio' / f'{path.stem}.mp3')
        dur = len(audio) / sr
        starts = [r[1] for r in timed]
        spans = [(s, min(dur, starts[i + 1] if i + 1 < len(starts) else dur)) for i, s in enumerate(starts)]
        X = features(audio, sr, spans)
        lab, sep = two_means(X)
        used_llm = False
        if a.alternate:
            lab = np.arange(len(timed)) % 2
            sep = 0.0
        elif a.llm or sep < 1.05:
            # Voices too alike for the audio to help: let the local LLM read
            # the dialogue and assign roles line by line (state 1 = doctor).
            llm = llm_roles([text for (_, _, _, text) in timed])
            if llm is not None:
                lab = llm; used_llm = True
            else:
                cues = np.array([line_cue(text) for (_, _, _, text) in timed])
                lab = viterbi(X, lab, sep, cues, p_switch=0.85)
        else:
            cues = np.array([line_cue(text) for (_, _, _, text) in timed])
            lab = viterbi(X, lab, sep, cues, p_switch=0.85)
        # which state is the doctor: the decoder already put the doctor in
        # state 1 (cue-weighted); in --alternate mode decide by total cue score
        score = {0: 0.0, 1: 0.0}
        for (i, t, tag, text), k in zip(timed, lab):
            if tag in ('D', 'P'):                          # human label wins, weight it heavily
                score[k] += 3.0 if tag == 'D' else -3.0
            score[k] += line_cue(text)
        doctor = max(score, key=score.get) if a.alternate else 1
        n_new = 0
        for (i, t, tag, text), k in zip(timed, lab):
            if tag is not None:
                continue
            m = MARK.match(lines[i])
            lines[i] = f'{lines[i][:m.end()].rstrip()} [{"D" if k == doctor else "P"}] {text}'
            n_new += 1
        if not a.dry:
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        flag = '   <- roles from the LLM (voices alike), check' if used_llm else ('' if sep >= 1.0 else ('   <- alternation fallback, check' if sep == 0.0 else '   <- low separation, check'))
        print(f'{tid}: {n_new} lines tagged, separation {sep:.2f}, doctor cluster cue score {score[doctor]:+.1f} vs {score[1 - doctor]:+.1f}{flag}')


if __name__ == '__main__':
    main()
