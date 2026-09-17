"""Optional replay recording through Oscar's Survival Lab recorder, when available.

The recorder lives in Oscar's research checkout; the path can be overridden with
TRAPPER_RECORDER_DIR. Replays saved under TRAPPER_REPLAY_DIR show up in the
live Survival Lab catalog automatically.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_RECORDER = Path.home() / 'Github_repos/Projects/nordic-ai-cup-2026/survival/debugger'
DEFAULT_REPLAYS = Path.home() / 'Github_repos/Projects/nordic-ai-cup-2026/survival/results/trapper/replays'


def recorder_available():
    d = Path(os.environ.get('TRAPPER_RECORDER_DIR', DEFAULT_RECORDER))
    return (d / 'recorder.py').exists()


def make_recorder(env, title, policy, seed, scenario, notes='', native=False, every=1):
    d = Path(os.environ.get('TRAPPER_RECORDER_DIR', DEFAULT_RECORDER))
    if not (d / 'recorder.py').exists():
        return None
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from recorder import ReplayRecorder
    rec = ReplayRecorder(env, title=title, policy=policy, seed=seed, scenario=scenario, notes=notes,
                         native_render=native, every=every)
    rec.capture(force=True)
    return rec


def save_recorder(rec, name, reason='finished'):
    if rec is None:
        return None
    out = Path(os.environ.get('TRAPPER_REPLAY_DIR', DEFAULT_REPLAYS))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{name}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}-{os.getpid() % 10000:04d}.json.gz'
    rec.save(path, reason=reason)
    return path
