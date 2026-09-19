"""Fork a sealed campaign into a separately frozen, budgeted continuation.

No games or LLM calls are started. Only historical development evidence is
imported. Original files, final selections and holdout artifacts stay untouched.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


PATCH_FILES = ('scripts/research_io.py', 'scripts/research_support.py',
    'scripts/experiment_case.py', 'scripts/research_diagnostics.py',
    'scripts/optimize_policy.py', 'scripts/research_loop.py',
    'scripts/research_study.py', 'scripts/research_resilient.py',
    'tests/test_research_recovery.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--patches',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pod-created-at',required=True)
    args=parser.parse_args()
    parent,source,out=map(Path.resolve,(args.parent,args.source,args.out))
    if source.exists() or out.exists():
        raise ValueError('Use new source and campaign directories')
    old=json.loads((parent/'state.json').read_text())
    base_config=json.loads((parent/'config.json').read_text())
    if old['status'] not in ('complete','inconclusive','failed','paused'):
        raise ValueError('Parent campaign must be stopped')
    best_code=parent/'snapshots'/old['best']['snapshot']/'code'
    sys.path.insert(0,str(best_code))
    from scripts.research_support import copy_source,manifest,digest,assert_snapshot
    parent_hashes={name:hashlib.sha256((parent/name).read_bytes()).hexdigest()
        for name in ('state.json','config.json','protocol.json','best.json','journal.md','final-selection.json')
        if (parent/name).exists()}
    files=json.loads((best_code.parent/'manifest.json').read_text())
    assert_snapshot(best_code,files)
    copy_source(best_code,source,files)
    for name in PATCH_FILES:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((args.patches/name).read_bytes())
    # Policy bytes must remain exactly the accepted policy's bytes.
    initial=manifest(source)
    assert all(initial.get(name)==value for name,value in files.items() if name.startswith('models/'))
    config=copy.deepcopy(base_config)
    config['budget'].update(total_usd=40,reserve_usd=5,external_spend_usd=10,
        hourly_rate_usd=.985,max_hours=24,agent_call_reserve_usd=0)
    config['schedule'].update(major_generations=8,require_max_generations=True,
        subgenerations=4,min_subgenerations=2,patience=2,sub_search_seconds=1800,
        comparison_seconds=5400,bo_seconds=3600,final_reserve_seconds=14400)
    config['agent'].update(executable='/root/.local/bin/codex',model='gpt-6-astra',max_calls=48)
    config['search'].update(focused_trials=8,bo_trials=24)
    config['storage'].update(campaign_gib=70,final_reserve_gib=8,minimum_free_gib=3)
    start_major=old['major']
    panels={}
    cursor=200000
    for major in range(start_major,9):
        for suffix in ('1','2','3','4','major'):
            panels[f'g{major}.{suffix}']=dict(screen_seeds=list(range(cursor,cursor+4)),
                comparison_seeds=list(range(cursor+4,cursor+12)),
                major_seeds=list(range(cursor+4,cursor+16)))
            cursor+=16
    first=panels[f'g{start_major}.1']
    config['evaluation'].update(first)
    config['evaluation'].update(seed_panels=panels,independent_confirmation=True,
        retired_holdout_seeds=base_config['evaluation']['holdout_seeds'],
        holdout_seeds=list(range(900000,900016)),case_timeout_seconds=7200,workers=16)
    config['recovery']=dict(parent_campaign=str(parent),case_retries=2,step_retries=2,
        coordinator_retries=5,parent_hashes=parent_hashes,
        historical_spend_allowance_usd=10,started_pod_at=args.pod_created_at)
    config_path=args.patches/'recovery-config.json'
    config_path.write_text(json.dumps(config,indent=2))
    subprocess.run([sys.executable,'-B',str(source/'scripts/research_loop.py'),'prepare',
        '--out',str(out),'--config',str(config_path)],check=True,cwd=source)
    state=json.loads((out/'state.json').read_text())
    # Freeze the original baseline under the same repaired harness, independently
    # of the accepted policy. The old snapshots themselves are never modified.
    original_old=parent/'snapshots'/old['original']['snapshot']/'code'
    original_files=json.loads((original_old.parent/'manifest.json').read_text())
    assert_snapshot(original_old,original_files)
    draft=out/'migration-original'
    copy_source(source,draft,initial)
    for name in list(manifest(draft)):
        if name.startswith('models/') and name not in original_files:
            (draft/name).unlink()
    for name in original_files:
        if name.startswith('models/'):
            target=draft/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((original_old/name).read_bytes())
    original_new=manifest(draft);original_id=digest(original_new)
    destination=out/'snapshots'/original_id/'code'
    copy_source(draft,destination,original_new)
    (destination.parent/'manifest.json').write_text(json.dumps(original_new,indent=2))
    def ref(old_ref,snapshot):
        value=dict(snapshot=snapshot,config=old_ref['config'],baseline=old_ref['baseline'])
        return dict(value,id=digest(value),label=old_ref['label']+' (recovered)')
    best=ref(old['best'],state['initial_snapshot'])
    original=ref(old['original'],original_id)
    state.update(status='prepared',stage='sub',major=start_major,sub=1,
        best=best,original=original,major_start_id=best['id'],
        started_at=datetime.fromisoformat(args.pod_created_at.replace('Z','+00:00')).timestamp(),
        agent_calls=old['agent_calls'],agent_reserved_usd=0,
        events=copy.deepcopy(old['events']),last_upstream=old.get('last_upstream'))
    state['events']['recovery-20260919']=dict(kind='authorized continuation',time=time.time(),details=dict(
        parent=str(parent),parent_hashes=parent_hashes,previous_best=old['best'],
        recovered_best=best,original=original,policy_bytes_unchanged=True,
        note='Recovery uses new infrastructure hashes and new seed panels. Earlier scores remain historical, not fresh validation. Old final assessments are sealed.'))
    (out/'state.json').write_text(json.dumps(state,indent=2))
    (out/'best.json').write_text(json.dumps(best,indent=2))
    (out/'previous-best.json').write_text(json.dumps(old['best'],indent=2))
    (out/'journal.md').write_bytes((parent/'journal.md').read_bytes())
    prompt=out/'research_agent_prompt.md'
    prompt.write_text(prompt.read_text()+f'\n\nRecovery continuation: historical development journal and artifacts are at {parent}. '
        'Never read that campaign\'s final-selection outcomes, final-report files, steps/final-holdout, or any case for retired seeds 1001 through 1016. '
        'The current context supplies a preassigned round panel, independent confirmation maps and remaining budget. '
        'Prioritize successful predator capture and sustained containment, bait continuity, worker feeding and measured policy CPU. '
        'Use the Lucas mirror at /workspace/lucas-bridge/upstream.git and its current.json receipt; do not assume it is live beyond that receipt. '
        'The three completed satellite studies are at /workspace/research-capture-fleet. Treat their old-map findings as hypotheses requiring new confirmation. '
        'The target is major generation 8 with adaptive stopping only within each major; a weak round does not cancel later major reviews.\n')
    guidance=out/'operator-guidance';guidance.mkdir()
    for name in ('astra-xhigh.receipt.json',):
        candidate=parent/'operator-guidance'/name
        if candidate.exists():shutil.copyfile(candidate,guidance/name)
    subprocess.run(['git','init',str(source)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(source),'remote','add','origin','/workspace/lucas-bridge/upstream.git'],check=True)
    for name,checksum in parent_hashes.items():
        assert hashlib.sha256((parent/name).read_bytes()).hexdigest()==checksum
    receipt=dict(parent=str(parent),out=str(out),source=str(source),major=start_major,
        parent_hashes=parent_hashes,previous_best=old['best']['id'],best=best['id'],
        initial_snapshot=state['initial_snapshot'],original_snapshot=original_id,
        policy_bytes_unchanged=True,holdout_outcomes_read=False,prepared_at=time.time())
    (out/'migration.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()
