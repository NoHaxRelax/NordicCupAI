"""Mandatory, uniquely named replays for every water/gap simulation."""
from common import ROOT, OUT, EXPERIMENT_HASHES
import hashlib
import json
import re
import sys
import uuid

sys.path.insert(0, str(ROOT / 'debugger'))
from recorder import ReplayRecorder


def start_recording(env, case, *, policy, notes, demonstration=None, reproduction=None):
    version = hashlib.sha256(json.dumps(EXPERIMENT_HASHES, sort_keys=True).encode()).hexdigest()
    seed = case.get('seed', 173)
    label = (demonstration or policy).replace('.replay.json', '').replace('.gz', '')
    label = re.sub(r'[^a-zA-Z0-9_-]+', '-', label).strip('-')
    run_id = f'{label}-v{version[:8]}-seed{seed}-{uuid.uuid4().hex[:12]}'
    path = OUT / 'replays' / f'{run_id}.json.gz'
    details = ', '.join(f'{k}={case[k]}' for k in
        ('site_index', 'gap', 'length', 'heading_offset', 'heading', 'relay', 'predators', 'comparison', 'release_s')
        if k in case)
    title = f'{policy} | seed {seed}' + (f' | {details}' if details else '')
    if reproduction:
        title = f"NEW reproduction: {reproduction['source_name']} row {reproduction['row_index'] + 1} | {title}"
        notes += ' New local rerun, not recovered historical footage. ' + reproduction['fidelity_note']
    notes += (' Native map, biomes, trees and food retained.' if case.get('native')
              else ' Constructed geometry and biome fixture.')
    if case.get('orchard'):
        notes += ' Six mature trees are arranged; no mature fruit is injected.'
    notes += (f" Requested horizon {case['seconds']} seconds; early termination may occur."
              if 'seconds' in case else ' Bounded physical survey; stops after route completion or loss.')
    if policy != 'Biome transect scout':
        notes += (' Additional native predator spawning is enabled.' if case.get('natural_predator_spawns')
                  else ' Additional predator spawning is disabled.')
    recorder = ReplayRecorder(env, title=title, policy=policy, seed=seed,
        every=10, scenario='generated' if case.get('native') else 'controlled',
        notes=notes, policy_sha256=version, native_render=bool(demonstration))
    recorder.meta.update(run_id=run_id, case=case, experiment_sha256=EXPERIMENT_HASHES,
                         recording_kind='new_reproduction' if reproduction else 'original_run')
    if reproduction:
        recorder.meta['reproduction'] = reproduction
    recorder.capture(force=True)
    return recorder, path


def finish_recording(recorder, path, *, reason):
    return dict(path=str(path.relative_to(ROOT)), **recorder.save(path, reason=reason))
