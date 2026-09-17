"""Mandatory, versioned replay publication for every colony experiment.

Uses the existing read-only recorder. No policy or simulator state is changed.
Historical backfills are explicitly new reproductions, never recovered frames.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'results/colony_banishment'


def suffix():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid4().hex[:8]


def save_json(data,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        os.link(temporary,path) # no-clobber atomic publication
    finally:temporary.unlink(missing_ok=True)
    return path


def save_summary(data,path):
    path=Path(path)
    if path.exists():path=path.with_name(path.stem+'-'+suffix()+path.suffix)
    save_json(data,path)
    print('Saved run summary: '+str(path),flush=True)
    return path


def start_recording(env, *, scenario, seed, mode, horizon, settings, version,
                    policy_hash, harness_hash, strategy_harness_hash, request=None):
    from recorder import ReplayRecorder
    options=dict(request) if isinstance(request,dict) else {}
    source=options.get('reproduction_of')
    hint=options.get('label') or (str(request) if isinstance(request,str) else 'run')
    slug=re.sub(r'[^a-zA-Z0-9_-]+','-',hint).strip('-')[:65]
    identity=f'colony-v{version}-{scenario}-seed{seed}-{mode}-{slug}-{suffix()}'
    path=OUT/'replays'/f'{identity}.json.gz'
    privilege='True agent poses supplied; predator energy/rest still hidden.' if mode.startswith('oracle') else 'Only ordinary cached observation DTOs and simulation time supplied.'
    setup='Default generated map and initial resources.' if scenario=='generated' else 'Arranged geometry and prepared finite orchard: 12 productive trees, 28 finite fruits, four 150-energy founders, scheduled predator arrivals at 20 and 100 seconds.'
    provenance=('NEW REPRODUCTION of '+source+'; original missing frames cannot be recovered and outcomes may differ. ') if source else ''
    notes=provenance+setup+' '+privilege+f' Horizon {horizon:g}s. Original movement, energy, food growth/rot, tree turnover, births and stochastic predator spawning. Local only.'
    recorder=ReplayRecorder(env,title=f"Colony v{version} | {scenario} | seed {seed} | {mode}"+(' | reproduction '+source if source else ' | '+identity[-8:]),
        policy=f'colony v{version} {mode}',seed=seed,scenario=scenario,every=int(options.get('every',10)),
        notes=notes,policy_sha256=policy_hash,native_render=bool(options.get('native_render',False)))
    recorder.meta.update(run_id=identity,policy_version=version,harness_sha256=harness_hash,
        strategy_harness_sha256=strategy_harness_hash,scenario_parameters=options.get('scenario_parameters'),
        controller_settings=settings,horizon=horizon,reproduction_of=source,
        original_result_sha256=options.get('original_result_sha256'),
        provenance_kind='new_reproduction' if source else 'original_recorded_run')
    recorder.capture(force=True)
    return recorder,path


def decisions_for_viewer(decisions):
    return {aid:({'rule':value,'detail':''} if isinstance(value,str) else value)
            for aid,value in decisions.items()}


def finish_recording(recorder,path,result):
    result['replay']=str(path.relative_to(ROOT.parent))
    result['run_id']=recorder.meta['run_id']
    result['reproduction_of']=recorder.meta.get('reproduction_of')
    result['policy_version']=recorder.meta['policy_version']
    result['recording_summary']=recorder.save(path,reason='extinction' if result['alive']==0 else 'horizon reached')
    metrics_path=OUT/'run-results'/f"{recorder.meta['run_id']}.json"
    result['result_file']=str(metrics_path.relative_to(ROOT.parent))
    save_json(result,metrics_path)
    return result
