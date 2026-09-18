"""Always-on, immutable replay and metrics publication for local relay research."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/sprinter_relay'


def unique_id(label):
    slug = re.sub(r'[^a-zA-Z0-9_-]+','-',label).strip('-')[:100]
    return f'{slug}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:10]}'


def start_replay(env, *, parameters, policy_sha256, title=None, every=5,
                 native_render=False, reproduction=None, assumptions=None):
    from recorder import ReplayRecorder
    policy = parameters.get('policy','fixture')
    seed = parameters.get('seed')
    version = policy_sha256[:10]
    label = f'{policy}-v{version}-seed{seed}'
    run_id = unique_id(label)
    if reproduction:
        description = f"NEW REPRODUCTION of {reproduction['source_case']}"
    else:
        description = title or f'Relay {policy}, E{parameters.get("energy")}, seed {seed}'
    notes = ('Arranged 1600x1200 clear forest with physical boundaries; assigned bait IDs. '
             'Policy uses cached observations and odometry, not true geometry or predator energy/rest. '
             f'Initial bait energy {parameters.get("energy")}; food={parameters.get("food")}; '
             f'horizon={parameters.get("seconds")} seconds. '
             'Native movement, energy, sensing, food, aging and births; no hosted calls.')
    if assumptions:
        notes = assumptions
    if reproduction:
        notes += (' New local reproduction, NOT recovered footage of the original metrics-only run. '
                  + reproduction.get('limitations',''))
    recorder = ReplayRecorder(env,title=description,policy=f'{policy} / {version}',seed=seed,
        every=every,scenario='arranged sprinter relay',notes=notes,
        policy_sha256=policy_sha256,native_render=native_render)
    recorder.meta.update(run_id=run_id,experiment_parameters=parameters,
                         recording_kind='new_reproduction' if reproduction else 'original_recording',
                         reproduction=reproduction)
    recorder.capture(force=True)
    return recorder, OUT/'replays'/(run_id+'.json.gz')


def finish_replay(recorder, path, result, *, reason='completed local experiment'):
    recorder.save(path,reason=reason,overwrite=False)
    result['replay'] = str(path.relative_to(OUT))
    result['run_id'] = recorder.meta['run_id']
    result['recording_kind'] = recorder.meta['recording_kind']
    result['reproduction'] = recorder.meta.get('reproduction')
    folder = OUT/'runs'
    folder.mkdir(parents=True,exist_ok=True)
    metric_path = folder/(result['run_id']+'.metrics.json')
    with metric_path.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    result['metrics_file'] = str(metric_path.relative_to(OUT))
    return result


def save_batch(name, payload):
    folder = OUT/'batches'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/(unique_id(name)+'.json')
    with path.open('x') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return path
