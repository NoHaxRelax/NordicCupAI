"""Snapshots of a running game (engine + policy) taken at interesting moments, replayable in
seconds instead of simulating whole games. pygame surfaces are dropped (they are only for drawing).
"""
from __future__ import annotations

import copyreg, gzip, json, pickle, time
from pathlib import Path

import pygame

copyreg.pickle(pygame.surface.Surface, lambda s: (type(None), ()))


def snapshot(env, policy, meta: dict, path: Path):
    """Write env + policy + meta to ``path`` (.pkl.gz) and a sidecar .json with meta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # the game runner monkeypatches engine methods with local closures (death accounting): drop
    # those instance attributes for the pickle and put them back afterwards
    patched = {k: v for k, v in list(env.__dict__.items()) if callable(v)}
    for k in patched:
        del env.__dict__[k]
    try:
        with gzip.open(path, 'wb', compresslevel=3) as f:
            pickle.dump(dict(env=env, policy=policy, meta=meta), f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        env.__dict__.update(patched)
    path.with_suffix('').with_suffix('.json').write_text(json.dumps(meta, indent=1, default=str))


def restore(path: Path):
    with gzip.open(Path(path), 'rb') as f:
        d = pickle.load(f)
    return d['env'], d['policy'], d['meta']
