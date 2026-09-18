"""Reproduce saved metrics-only cases without altering the original evidence.

Run groups in separate processes if desired. Original summary files are immutable
inputs. Each reproduced case gets an individual receipt, replay and metric file.
No claim is made that new recordings recover missing original frames.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
OUT = BASE.parents[1]/'results/sprinter_relay'
PHASES = {
 'pilot':('run_exploratory','controller_before_relay','Closest preserved pre-relay controller. Original full source hash is unavailable; radial/MPC behavior is retained.'),
 'iterate':('run_exploratory','controller_before_relay','Closest preserved pre-relay controller; full original source hash differs. Original sweep parameters are retained.'),
 'orbit-screen':('run_exploratory','controller_before_relay','Exact preserved controller hash; exploratory runner instrumentation was subsequently extended.'),
 'relay-screen':('run_exploratory','controller_stage2','Exact preserved controller hash; exploratory instrumentation was subsequently extended.'),
 'heldout-invalid-single-baseline':('run_exploratory','controller_stage3','Excluded historical baseline. Closest preserved controller. Original orbit_single label dispatched MPC; reproduce it explicitly as single/MPC.'),
 'heldout':('run','controller','Exact recorded controller hash. Recorder added; final metrics are measured again, not copied.'),
 'workers':('run','controller','Exact recorded controller hash. Recorder added; final metrics are measured again, not copied.'),
 'multiple':('run','controller','Exact recorded controller hash. Recorder added; final metrics are measured again, not copied.'),
 'renewal-screen':('run','controller','Exact recorded controller hash. Recorder added; final metrics are measured again, not copied.'),
}
GROUPS={'historical':['pilot','iterate','orbit-screen','relay-screen'],
        'final':['heldout','workers','multiple','renewal-screen'],
        'excluded':['heldout-invalid-single-baseline']}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=[*GROUPS,'all'],default='all')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    phases=list(PHASES) if args.group=='all' else GROUPS[args.group]
    receipts=OUT/'backfill-receipts';receipts.mkdir(exist_ok=True)
    completed=0
    for phase in phases:
        source=OUT/(phase+'.json');source_hash=sha(source)
        payload=json.loads(source.read_text())
        module_name,controller_name,limitations=PHASES[phase]
        runner=importlib.import_module(module_name)
        controller=importlib.import_module(controller_name)
        runner.RelayController=controller.RelayController
        if hasattr(controller,'MultiRelayController'):runner.MultiRelayController=controller.MultiRelayController
        for index,old in enumerate(payload['runs']):
            key=f'{phase}-{index:03d}'
            receipt_path=receipts/(key+'.json')
            if receipt_path.exists():
                receipt=json.loads(receipt_path.read_text())
                if (OUT/receipt['replay']).exists():continue
                raise RuntimeError(f'Missing replay referenced by {receipt_path}')
            if old.get('replay') and (OUT/old['replay']).exists():continue
            reproduction=dict(source_case=f'{phase}[{index}]',source_file=str(source.relative_to(OUT)),
                source_file_sha256=source_hash,source_row_index=index,
                source_controller_sha256=old.get('controller_sha256',payload.get('controller_sha256')),
                selected_controller=controller_name,selected_controller_sha256=sha(Path(controller.__file__)),
                limitations=limitations,original_parameters=old['parameters'])
            parameters=dict(old['parameters'])
            if phase=='heldout-invalid-single-baseline' and parameters['policy']=='orbit_single':
                parameters['policy']='single'
            row=runner.run(**parameters,reproduction=reproduction,record_every=5)
            receipt=dict(source_case=reproduction['source_case'],source_file_sha256=source_hash,
                replay=row['replay'],metrics_file=row['metrics_file'],run_id=row['run_id'],
                recording_kind='new_reproduction',controller=controller_name,limitations=limitations,
                original_retention=old.get('active_retention'),reproduced_retention=row.get('active_retention'),
                original_duration=parameters['seconds'],reproduced_duration=parameters['seconds'])
            with receipt_path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
            assert sha(source)==source_hash,'Original metrics changed during backfill'
            completed+=1
            print(json.dumps(receipt),flush=True)
            if args.limit and completed>=args.limit:return

if __name__=='__main__':main()
