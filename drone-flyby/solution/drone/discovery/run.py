#!/usr/bin/env python3
"""Finite validation discovery. No model calls; at most 100 reserved validations."""
import argparse
import csv
import fcntl
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT,DATA,PRIVATE,atomic,read,now,digest,predictions
from engine import Search,coverage_ok,missing_frames,relevant_errors
from planner import build

TERMINAL={'budget_exhausted','batch_complete','paused','needs_attention'}
URL_FILE=PRIVATE/'discovery-endpoint-url'
PORT=9055


def alive(pid):
    try:os.kill(int(pid),0);return True
    except PermissionError:return True  # EPERM still establishes that the PID exists.
    except (OSError,TypeError,ValueError):return False


def health():
    s=read(DATA/'status.json',{'status':'not_started'})
    s['checked_at']=now();s['worker_alive']=alive(s.get('worker_pid'))
    s['heartbeat_age_seconds']=round(time.time()-s.get('heartbeat_epoch',0),1) if s.get('heartbeat_epoch') else None
    s['healthy']=s.get('status') in (TERMINAL-{'needs_attention'}) or (s.get('status')!='needs_attention' and s['worker_alive'] and s['heartbeat_age_seconds'] is not None and s['heartbeat_age_seconds']<120)
    return s


class Worker:
    def __init__(self):
        self.children=[];self.stop=False;self.runtime={};self.last_write=0;self.mode='starting';self.stage='Starting';self.error=None
        state=read(DATA/'state.json')
        if state:self.search=Search(DATA,state)
        else:
            roots,seeds=build();self.search=Search.create(DATA,roots,seeds,100)
        self.s=self.search.state
        self.s.setdefault('started_at',now())
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        self.publish()

    def event(self,message):
        self.s['events'].append({'at':now(),'message':message})
        self.s['events']=self.s['events'][-100:];self.search.save()

    def publish(self,force=False):
        if not force and time.time()-self.last_write<2:return
        self.last_write=time.time();p=self.s.get('pending');details=None
        if p:
            receipts=list((DATA/'receipts'/p['query_id']).glob('*.json'))
            targets=set(p.get('target_frame_ids',[]))
            received={int(f.stem.rsplit('-',1)[-1]) for f in receipts}
            details={'query_id':p['query_id'],'phase':p['phase'],'label':p['node']['label'],
                     'hypotheses':p['node']['hypothesis_count'],'received_frames':len(receipts),
                     'received_target_frames':len(targets&received),
                     'target_frames':p.get('target_frames',0),'submitted_at':p.get('submitted_at')}
        completed=self.s['runs_completed'];durations=self.s.get('durations',[])
        seconds=sum(durations)/len(durations) if durations else 90
        out={'schema':1,'status':self.mode,'stage':self.stage,'error':self.error,
             'worker_pid':os.getpid(),'heartbeat_at':now(),'heartbeat_epoch':time.time(),
             'started_at':self.s['started_at'],'runs_used':self.s['runs_reserved'],
             'runs_completed':completed,'max_runs':self.s['max_runs'],
             'confirmed_total':len(self.s['seeds']),'confirmed_new':len(self.s['seeds'])-self.s['initial_seed_count'],
             'classes_confirmed':sorted({x['class'] for x in self.s['seeds']}),
             'roots_screened':sum(x['status'] in ('positive_group','seed_found','no_match_in_tested_hypotheses') for x in self.s['roots'].values()),
             'roots_total':len(self.s['roots']),'positive_groups':sum(x['status'] in ('positive_group','seed_found') for x in self.s['roots'].values()),
             'classes_with_positive_entry_groups':sorted(x['label'].split(':')[0] for x in self.s['roots'].values() if x['kind']=='entry_scan' and x['status'] in ('positive_group','seed_found')),
             'remaining_groups':len(self.s['screen'])+len(self.s['work'])+len(self.s.get('retries',[]))+len(self.s.get('unresolved',[]))+(1 if p else 0),
             'incomplete_attempts':len(self.s.get('incomplete_attempts',[])),
             'deferred_groups':len(self.s.get('unresolved',[])),
             'current':details,'events':self.s['events'][-8:],
             'estimated_seconds_to_budget':round(max(0,self.s['max_runs']-completed)*seconds),
             'coverage_note':'Reference-sized dense local grids plus sampled initial-frame and entry searches. Zero means no match among tested boxes, not object absence. Frames 1–4 lack complete native pixels.',
             'search_schedule':self.s.get('schedule','screen_first'),
             'all_objects_found':False,'runtime':{k:v for k,v in self.runtime.items() if k.endswith('_pid')}}
        atomic(DATA/'status.json',out)

    def export(self):
        atomic(DATA/'confirmed-seeds.json',{'source_dimensions':[3840,2160],
            'updated_at':now(),'complete':False,'annotations':self.s['seeds'],
            'limitations':['Only score-confirmed seed boxes are included. A match establishes IoU >= 0.50, not exact organizer coordinates.',
                           'Seeds are annotations, not counts of distinct physical objects. Seven prior boxes belong to the same launcher track.',
                           'Unlabelled image regions must not be treated as certified background. Frames 1–4 lack complete 4K pixels.']})
        buf=io.StringIO();writer=csv.writer(buf);writer.writerow(['seed_id','frame','class','x1','y1','x2','y2','evidence','receipt'])
        for a in self.s['seeds']:writer.writerow([a['seed_id'],a['frame'],a['class'],*a['bbox_source_xyxy'],a['evidence'],a.get('receipt',a.get('source',''))])
        tmp=DATA/'confirmed-seeds.csv.tmp';tmp.write_text(buf.getvalue());tmp.replace(DATA/'confirmed-seeds.csv')

    def pause(self,seconds,check_children=True):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.publish()
            if check_children:
                for kind,proc in self.children:
                    if kind!='awake' and proc.poll() is not None:raise RuntimeError(kind+' process exited; queued work will not be resubmitted')
            time.sleep(min(1,max(0,until-time.monotonic())))

    def portal(self,action,path,uuid=None):
        command=[sys.executable,str(ROOT/'drone/portal.py'),action,'--url-file',str(URL_FILE),'--output',str(path)]
        if uuid:command.extend(['--uuid',uuid])
        # Do not expose raw stdout/stderr: API responses can contain capabilities.
        proc=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        limit=time.monotonic()+55
        while proc.poll() is None:
            self.publish();time.sleep(.3)
            if time.monotonic()>limit:proc.kill();proc.wait();raise RuntimeError('Official API request timed out: '+action)
        if proc.returncode:raise RuntimeError('Official API request failed: '+action+'; inspect the saved sanitized HTTP receipt')
        data=read(path)
        if data is None:raise RuntimeError('Official API response is not readable: '+action)
        return data

    def get_with_retry(self,action,path,uuid):
        for attempt in range(4):
            try:return self.portal(action,path,uuid)
            except RuntimeError:
                if attempt==3:raise
                self.stage='Retrying result lookup (no new submission)';self.pause(5*(attempt+1),False)

    def activate(self,plan):
        atomic(DATA/'active-plan.json',plan)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            h=read(DATA/'endpoint-health.json',{})
            if h.get('loaded_plan_hash')==plan['plan_hash'] and h.get('loaded_query')==plan['query_id']:return
            self.pause(.4)
        raise RuntimeError('Recorder did not acknowledge the current plan')

    def start_services(self):
        self.stage='Starting callback recorder and temporary relay';self.publish(True)
        # Dedicated port; never replace or terminate unrelated local services.
        import socket
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',PORT))==0:raise RuntimeError('Discovery callback port is already occupied; no process was stopped')
        atomic(DATA/'active-plan.json',{'query_id':'preflight','predictions_by_frame':{},'plan_hash':digest({})})
        log=open(DATA/'endpoint.log','ab',buffering=0)
        recorder=subprocess.Popen([sys.executable,str(ROOT/'drone/discovery/endpoint.py'),'--port',str(PORT),
            '--route-file',str(PRIVATE/'route'),'--plan-file',str(DATA/'active-plan.json'),
            '--receipt-dir',str(DATA/'receipts'),'--health-file',str(DATA/'endpoint-health.json')],stdout=log,stderr=log)
        log.close();self.children.append(('recorder',recorder));self.runtime['recorder_pid']=recorder.pid
        self.activate(read(DATA/'active-plan.json'))
        relay_path=PRIVATE/'discovery-tunnel.log'
        fd=os.open(relay_path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);os.chmod(relay_path,0o600)
        relay=subprocess.Popen([str(PRIVATE/'cloudflared'),'tunnel','--url',f'http://127.0.0.1:{PORT}',
            '--no-autoupdate','--protocol','http2'],stdout=fd,stderr=fd)
        os.close(fd);self.children.append(('relay',relay));self.runtime['relay_pid']=relay.pid
        deadline=time.monotonic()+75;url=None
        while time.monotonic()<deadline:
            match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',relay_path.read_text(errors='ignore'))
            if match:url=match.group(0);break
            self.pause(1)
        if not url:raise RuntimeError('Temporary relay did not publish an endpoint')
        route=(PRIVATE/'route').read_text().strip()
        fd=os.open(URL_FILE,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);os.chmod(URL_FILE,0o600)
        with os.fdopen(fd,'w') as f:f.write(url+'/'+route+'/predict\n')
        atomic(DATA/'runtime.json',{**self.runtime,'worker_pid':os.getpid(),'started_at':now()})
        self.pause(3)
        for attempt in range(3):
            try:
                verified=self.portal('verify',DATA/'preflight.json')
                # This endpoint raises on schema errors. A clean verify is not a
                # validation submission and never consumes the 100-run budget.
                if verified.get('errors') or not verified.get('can_connect') or not verified.get('can_predict') or verified.get('response_status_code')!=200:
                    raise RuntimeError('Official verification did not confirm a working callback')
                break
            except RuntimeError:
                if attempt==2:raise
                self.pause(5)
        self.event('Callback verified. Standalone discovery is ready.')
        if Path('/usr/bin/caffeinate').exists():
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.children.append(('awake',awake))

    def finish_pending(self):
        p=self.s['pending'];folder=DATA/'queries'/p['query_id'];folder.mkdir(parents=True,exist_ok=True)
        if p['phase']=='prepared':
            plan={'query_id':p['query_id'],'predictions_by_frame':predictions(self.search.items(p['node']))}
            plan['plan_hash']=digest(plan['predictions_by_frame']);atomic(folder/'plan.json',plan)
            self.activate(plan)
            p.update(phase='submitting',target_frames=len(plan['predictions_by_frame']),
                     target_frame_ids=list(map(int,plan['predictions_by_frame'])),submitted_at=now())
            self.search.save()  # Reserve and journal BEFORE any external POST.
            queued=self.portal('validate',folder/'queue.json')
            p.update(phase='queued',uuid=queued['queued_attempt_uuid']);self.search.save()
        if p['phase']=='submitting':
            # A process may die after the API writes a receipt but before the
            # state transition. Recover that receipt; never repeat this POST.
            queued=read(folder/'queue.json',{})
            if not queued.get('queued_attempt_uuid'):
                raise RuntimeError('Submission outcome is uncertain. Inspect the official queue before any restart; no automatic resubmission.')
            p.update(phase='queued',uuid=queued['queued_attempt_uuid']);self.search.save()
        self.mode='running';self.stage='Screening groups' if self.s['roots'][p['node']['root']]['status']=='pending' else 'Isolating a positive box'
        if p['node'].get('recovery_attempts'):self.stage='Automatically retrying an incomplete query'
        deadline=time.monotonic()+20*60
        result=read(folder/'result.json')
        while result is None:
            queue=self.get_with_retry('queue',folder/'progress.json',p['uuid'])
            status=queue.get('status');p['phase']=status;self.search.save();self.publish(True)
            if status in ('done','completed','finished'):
                result=self.get_with_retry('result',folder/'result.json',p['uuid']);break
            if status not in ('queued','in_progress','running'):raise RuntimeError('Unexpected official queue state; inspect this saved query')
            if time.monotonic()>deadline:raise RuntimeError('Existing query exceeded 20 minutes; it will not be submitted again')
            # A relay failure while running invalidates the attempt, but still
            # poll the existing UUID to completion before any future submission.
            self.pause(10,False)
        plan=read(folder/'plan.json')
        receipts=[read(f,{}) for f in (DATA/'receipts'/p['query_id']).glob('*.json')]
        ok,reason=coverage_ok(plan,receipts,result)
        score=result.get('score')
        if not isinstance(score,(float,int)) or not math.isfinite(score) or score<0:raise RuntimeError('Invalid score in evaluator result')
        if not ok:
            gaps=missing_frames(plan,receipts)
            # Wrong-plan or wrong-count receipts can mean unplanned predictions
            # were evaluated. They cannot support narrowing a zero to missing frames.
            matching=all(r.get('plan_hash')==plan['plan_hash'] and
                         r.get('annotation_count')==len(plan['predictions_by_frame'].get(str(r.get('frame')),[])) for r in receipts)
            clean_zero=score==0 and not relevant_errors(plan,result) and matching
            decision=self.search.recover_incomplete(score,gaps,str((folder/'result.json').relative_to(ROOT)),reason,clean_zero)
            if decision['action']=='retry':
                detail=f'{len(gaps)} missing frames' if decision['scope']=='missing_frames' else 'the full affected group'
                self.stage='Recovering incomplete frame delivery'
                self.event(f'{decision["query_id"]}: incomplete; automatically retrying {detail} within the same budget')
            else:
                self.event(f'{decision["query_id"]}: repeated delivery failures; group kept unresolved while other searches continue')
            self.publish(True);return
        try:
            start=datetime.fromisoformat(result['started_at']);end=datetime.fromisoformat(result['finished_at'])
            self.s.setdefault('durations',[]).append((end-start).total_seconds()+7)
        except (KeyError,ValueError):pass
        qid=p['query_id'];count=len(self.s['seeds'])
        self.search.observe(score,str((folder/'result.json').relative_to(ROOT)))
        self.export()
        suffix='; confirmed a new seed' if len(self.s['seeds'])>count else ''
        self.event(f'{qid}: '+('positive' if score>0 else 'no match in tested boxes')+suffix)
        self.publish(True)

    def run(self):
        self.export()
        try:
            if self.s['pending'] and self.s['pending']['phase']!='prepared':
                # Recovery is read-only until the previous UUID is accounted for.
                self.finish_pending()
            if self.s['runs_reserved']>=self.s['max_runs']:
                self.mode='budget_exhausted';self.stage='100-run budget finished';return
            self.start_services();self.mode='running'
            while True:
                if self.stop or (DATA/'STOP').exists():self.mode='paused';self.stage='Paused between queries';break
                if self.s['pending']:self.finish_pending();continue
                node=self.search.next_node()
                if node is None:
                    self.mode='budget_exhausted' if self.s['runs_reserved']>=self.s['max_runs'] else 'batch_complete'
                    self.stage='Query budget finished' if self.mode=='budget_exhausted' else ('Prepared search finished; delivery gaps remain' if self.s.get('unresolved') else 'Prepared search finished')
                    self.event(self.stage+'. Unsearched areas remain unknown.');break
                self.search.reserve(node);self.finish_pending()
        except Exception as e:
            self.mode='needs_attention';self.stage='Stopped safely; checkpoint preserved'
            self.error=str(e) if isinstance(e,RuntimeError) else type(e).__name__+' in discovery worker'
            self.event(self.error)
        finally:
            for kind,proc in reversed(self.children):
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
            self.export();self.search.save();self.publish(True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','start','run','health','stop'])
    a=p.parse_args();DATA.mkdir(parents=True,exist_ok=True)
    if a.action=='health':print(json.dumps(health(),indent=2));return
    if a.action=='stop':(DATA/'STOP').write_text('Pause after the current query\n');print('Pause requested');return
    if a.action=='prepare':
        if (DATA/'state.json').exists():raise SystemExit('Existing journal preserved; prepare is not a reset command')
        roots,seeds=build();s=Search.create(DATA,roots,seeds,100)
        print(json.dumps({'groups':len(roots),'hypotheses':sum(x['hypothesis_count'] for x in roots),'existing_seeds':len(seeds),'budget':100}));return
    if a.action=='start':
        if (DATA/'STOP').exists():raise SystemExit('STOP is present; remove it only when resumption is authorized')
        lock=open(DATA/'worker.lock','a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Worker is already running')
        # Transfer this same lock descriptor across exec. No double-start gap.
        log=open(DATA/'worker.log','ab',buffering=0)
        env=dict(os.environ);env['DRONE_DISCOVERY_LOCK_FD']=str(lock.fileno())
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=ROOT,env=env,
            stdout=log,stderr=log,start_new_session=True,pass_fds=(lock.fileno(),))
        log.close();lock.close();print(json.dumps({'worker_pid':proc.pid,'budget':100,'dashboard':'http://127.0.0.1:9054/'}));return
    fd=os.environ.pop('DRONE_DISCOVERY_LOCK_FD',None)
    if fd is not None:lock=os.fdopen(int(fd),'a+')
    else:
        lock=open(DATA/'worker.lock','a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Worker is already running')
    worker=Worker();worker.run()

if __name__=='__main__':main()
