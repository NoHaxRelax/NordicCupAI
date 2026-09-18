"""Frozen held-out protocol. Writes only this task's results directory."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
from recording_support import save_summary
from run import HERE, OUT, POLICY_HASH, HARNESS_HASH, COMMIT, run_case, STRATEGY_HARNESS_SHA256


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['generated','fixtures','oracle']);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    freeze=json.loads((OUT/'heldout-v5-freeze.json').read_text())
    archived=HERE/'archives/run_v5_before_mandatory_recording.py.txt'
    assert freeze['policy_sha256']==POLICY_HASH and freeze['harness_sha256']==STRATEGY_HARNESS_SHA256,'Frozen code changed'
    assert hashlib.sha256(archived.read_bytes()).hexdigest()==STRATEGY_HARNESS_SHA256,'Archived strategy harness changed'
    jobs=[];settings=freeze['settings']
    if a.phase=='generated':
        for seed in freeze['heldout_generated_seeds']:
            for mode in ('control','banish','nursery'):
                replay=f'heldout-v5-generated-{seed}-{mode}.json.gz' if seed==201 and mode!='nursery' else None
                jobs.append(('generated',seed,mode,freeze['generated_horizon'],settings,replay))
    elif a.phase=='fixtures':
        for scenario in ('forest','obstacles','river'):
            for seed in freeze['heldout_fixture_seeds']:
                for mode in ('control','banish'):
                    replay=f'heldout-v5-{scenario}-{seed}-{mode}.json.gz' if scenario=='forest' and seed==21 or scenario=='obstacles' and seed==22 and mode=='banish' else None
                    jobs.append((scenario,seed,mode,freeze['fixture_horizon'],settings,replay))
    else:
        for scenario in ('forest','obstacles','river'):
            for seed in (21,22):
                for mode in ('oracle-control','oracle'):
                    jobs.append((scenario,seed,mode,300,settings,None))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run_case,jobs))
    data=dict(source_commit=COMMIT,protocol=freeze,runs=results,scope='Local held-out strategy evaluation; no external operations.',
        pairing_note='Same setup seeds; behavior changes consumption of shared RNG. Object-set iteration and floating geometry can also cause repeat divergence.')
    save_summary(data,OUT/f'heldout-v5-{a.phase}.json')


if __name__=='__main__':main()
