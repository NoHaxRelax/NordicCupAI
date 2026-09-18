"""Preflight, run a finite sequential GPU queue, or report its health."""
import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from common import read, release_errors, sha, verify_dataset, write


def utc():return dt.datetime.now(dt.timezone.utc)


def preflight(args):
    manifest=read(args.data/'manifest.json');release=read(args.release)
    report=dict(host=platform.node(),checked_at=utc().isoformat(),dataset=str(args.data),
                counts=manifest['counts'],coverage=manifest['coverage'],release_errors=release_errors(manifest,release))
    verify_dataset(args.data,manifest)
    report['dataset_hashes_verified']=True
    report['weights']={name:dict(exists=(args.weights/name).exists(),sha256=sha(args.weights/name) if (args.weights/name).exists() else None)
                       for name in ['yolo26x.pt','resnet50-11ad3fa6.pth']}
    try:
        import torch
        report['torch']=torch.__version__;report['ultralytics']=importlib.metadata.version('ultralytics')
        report['cuda']=torch.cuda.is_available();report['visible_gpus']=torch.cuda.device_count()
        if report['cuda']:
            report['gpu']=torch.cuda.get_device_name(0)
            report['vram_gib']=torch.cuda.get_device_properties(0).total_memory/2**30
    except ImportError:
        report['cuda']=False;report['runtime_missing']=True
    report['ready']=not report['release_errors'] and report.get('visible_gpus')==1 and all(w['exists'] for w in report['weights'].values())
    if args.output:write(args.output,report)
    print(json.dumps(report,indent=2))
    return 0 if report['ready'] else 2


def health(args):
    if not args.run.exists():
        print(json.dumps(dict(state='not_started',run=str(args.run))));return 0
    status=read(args.run/'status.json') if (args.run/'status.json').exists() else {'state':'unknown'}
    completed=[]
    for p in sorted(args.run.glob('*/result.json')):
        r=read(p)
        completed.append(dict(experiment=r['experiment'],checkpoint=r['checkpoint'],smoke=r.get('smoke',False),dev=r['dev']))
    status['results']=completed
    current=status.get('current_output')
    if current and (Path(current)/'progress.json').exists():status['progress']=read(Path(current)/'progress.json')
    status['lock_present']=(args.run/'queue.lock').exists() or (args.run.parent/'overnight-mypc.gpu.lock').exists()
    print(json.dumps(status,indent=2));return 0


def queue(args):
    manifest=read(args.data/'manifest.json')
    failures=release_errors(manifest,read(args.release))
    if failures:raise SystemExit('; '.join(failures))
    # One queue/GPU on mypc. Different HPC jobs may deliberately select disjoint experiments.
    args.run.mkdir(parents=True,exist_ok=True)
    lock=args.run.parent/'overnight-mypc.gpu.lock' if args.host=='mypc' else args.run/'queue.lock'
    try:lock.mkdir()
    except FileExistsError:raise SystemExit('Queue lock exists. Verify its host/PID before recovery; never launch a duplicate.')
    write(lock/'owner.json',dict(pid=os.getpid(),host=platform.node(),run=str(args.run.resolve()),started_at=utc().isoformat()))
    status=dict(state='starting',pid=os.getpid(),host=platform.node(),started_at=utc().isoformat(),completed=[],failed=[])
    try:
        verify_dataset(args.data,manifest)
        selected=args.experiments.split(',') if args.experiments else [s['id'] for s in manifest['config']['experiments']]
        allowed={s['id'] for s in manifest['config']['experiments']}
        if not selected or len(selected)!=len(set(selected)) or set(selected)-allowed:raise ValueError('Invalid experiment selection')
        for experiment in selected:
            release=read(args.release)
            if release_errors(manifest,release):
                status['state']='paused_by_release';break
            if utc()>=dt.datetime.fromisoformat(release['stop_starting_jobs_at']):
                status['state']='window_finished';break
            output=args.run/experiment
            if (output/'result.json').exists():
                saved=read(output/'runtime.json')
                if saved['identity']['manifest_sha256']!=sha(args.data/'manifest.json'):
                    raise ValueError('Completed run belongs to another dataset')
                status['completed'].append(experiment);continue
            if output.exists():
                status['failed'].append(dict(experiment=experiment,reason='Existing unfinished output requires inspected resume'))
                continue
            status.update(state='running',current=experiment,current_output=str(output.resolve()),updated_at=utc().isoformat())
            command=[sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--data',str(args.data.resolve()),
                     '--weights',str(args.weights.resolve()),'--output',str(output.resolve()),'--release',str(args.release.resolve()),
                     '--experiment',experiment,'--host',args.host]
            with (args.run/(experiment+'.out')).open('a') as stdout,(args.run/(experiment+'.err')).open('a') as stderr:
                process=subprocess.Popen(command,stdout=stdout,stderr=stderr)
                status['training_pid']=process.pid;write(args.run/'status.json',status)
                while process.poll() is None:
                    status['updated_at']=utc().isoformat();write(args.run/'status.json',status);time.sleep(10)
                if process.returncode==0 and (output/'result.json').exists():status['completed'].append(experiment)
                else:status['failed'].append(dict(experiment=experiment,exit_code=process.returncode))
            write(args.run/'status.json',status)
        else:status['state']='finished_with_failures' if status['failed'] else 'completed'
        status.pop('training_pid',None);status['finished_at']=utc().isoformat();write(args.run/'status.json',status)
    except BaseException as exc:
        status.update(state='needs_attention',error=str(exc),updated_at=utc().isoformat());write(args.run/'status.json',status)
        raise
    finally:
        # If interrupted with a live child, retain the lock so no follow-up starts a duplicate.
        child=locals().get('process')
        if child is None or child.poll() is not None:
            (lock/'owner.json').unlink(missing_ok=True);lock.rmdir()
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for name in ['preflight','queue']:
        q=sub.add_parser(name)
        for field in ['data','weights','release']:q.add_argument('--'+field,type=Path,required=True)
        if name=='preflight':q.add_argument('--output',type=Path)
        else:
            q.add_argument('--run',type=Path,required=True);q.add_argument('--host',choices=['mypc','hpc','runpod'],required=True)
            q.add_argument('--experiments',help='Comma-separated subset; default is all configured experiments')
    q=sub.add_parser('health');q.add_argument('--run',type=Path,required=True)
    args=p.parse_args();raise SystemExit(globals()[args.action](args))
