"""Reference-sized dense local grids and a held-out geometry check."""
import argparse
import fcntl
import math
import statistics
from collections import defaultdict
from pathlib import Path
from common import ROOT,DATA,W,H,atomic,read,hypothesis,iou,now


def observations():
    rows=[]
    for p in sorted((ROOT/'data/drone/reference/helsinki/annotations').glob('*.json')):
        d=read(p)
        for a in d['annotations']:
            b=a['bbox']
            if b[0]<=0 or b[1]<=0 or b[2]>=W-1 or b[3]>=H-1:continue
            rows.append({'frame':d['frame'],'class':a['object_id'],'box':b,
                         'cx':(b[0]+b[2])/2,'cy':(b[1]+b[3])/2,'w':b[2]-b[0],'h':b[3]-b[1]})
    return rows


def fit(rows):
    classes=defaultdict(list)
    for r in rows:classes[r['class']].append(r)
    out={}
    for label,values in classes.items():
        cy=statistics.mean(x['cy'] for x in values);den=sum((x['cy']-cy)**2 for x in values)
        shape={}
        for key in ('w','h'):
            mean=statistics.mean(x[key] for x in values)
            slope=sum((x['cy']-cy)*(x[key]-mean) for x in values)/den if den else 0
            shape[key]={'intercept':mean-slope*cy,'slope':slope,'minimum':min(x[key] for x in values),
                        'maximum':max(x[key] for x in values),'median':statistics.median(x[key] for x in values)}
        out[label]={'count':len(values),'shape':shape}
    return out


def size(model,label,cy):
    dimensions=[]
    for key in ('w','h'):
        p=model[label]['shape'][key]
        dimensions.append(max(.9*p['minimum'],min(1.1*p['maximum'],p['intercept']+p['slope']*cy)))
    return dimensions


def local_grid(frame,label,cx,cy,model):
    w,h=size(model,label,cy);unique={}
    for rotated in (False,True):
        for scale in (.9,1.1):
            pw,ph=(h*scale,w*scale) if rotated else (w*scale,h*scale)
            for ox in (-.5,-.25,0,.25,.5):
                for oy in (-.5,-.25,0,.25,.5):
                    x=cx+ox*pw;y=cy+oy*ph
                    b=[max(0,x-pw/2),max(0,y-ph/2),min(W,x+pw/2),min(H,y+ph/2)]
                    value=hypothesis(frame,label,b)
                    if value:unique[tuple(value['bbox_source_xyxy'])]=value
    return list(unique.values())


def benchmark():
    rows=observations();train=[r for r in rows if r['frame']%2==0];held=[r for r in rows if r['frame']%2==1]
    model=fit(train);scores=defaultdict(lambda:{'cases':0,'old_hits':0,'dense_hits':0,'minimum_dense_iou':1})
    for r in held:
        label=r['class']
        if label not in model:continue
        for ox in (-.5,-.25,0,.25,.5):
            for oy in (-.5,-.25,0,.25,.5):
                cx=r['cx']+ox*r['w'];cy=r['cy']+oy*r['h']
                dense=local_grid(r['frame'],label,cx,cy,model)
                old=[];p=model[label]['shape']
                for sx in (.55,.75,1,1.35,1.8):
                    for sy in (.55,.75,1,1.35,1.8):
                        w=p['w']['median']*sx;h=p['h']['median']*sy
                        v=hypothesis(r['frame'],label,[cx-w/2,cy-h/2,cx+w/2,cy+h/2])
                        if v:old.append(v)
                a=max((iou(r['box'],v['bbox_source_xyxy']) for v in old),default=0)
                b=max((iou(r['box'],v['bbox_source_xyxy']) for v in dense),default=0)
                s=scores[label];s['cases']+=1;s['old_hits']+=a>=.5;s['dense_hits']+=b>=.5;s['minimum_dense_iou']=min(s['minimum_dense_iou'],b)
    summary={key:sum(s[key] for s in scores.values()) for key in ('cases','old_hits','dense_hits')}
    report={'created_at':now(),'training_frames':'even reference frame numbers','held_out_frames':'odd reference frame numbers',
            'unclipped_training_boxes':len(train),'unclipped_held_out_boxes':len(held),'summary':summary,'classes':dict(scores),
            'method':'Shift the proposed center by up to half of the true width and height in 25 combinations; test whether any hypothesis overlaps ground truth at IoU >= 0.50.',
            'limitation':'Geometry-only calibration using repeated views of one reference instance per class. It does not measure object proposal recall or guarantee validation sizes, orientations or full-scene coverage.'}
    atomic(DATA/'grid-calibration.json',report);atomic(DATA/'size-model.json',fit(rows))
    return report


def apply():
    lock=open(DATA/'worker.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Pause the worker between queries before changing its schedule')
    s=read(DATA/'state.json')
    if s['pending']:raise RuntimeError('Pending query must finish before grid migration')
    if s.get('schedule')=='seed_first':raise RuntimeError('Grid migration is already applied')
    if not (DATA/'STOP').exists():raise RuntimeError('Expected explicit pause marker')
    report=benchmark();model=read(DATA/'size-model.json')
    atomic(DATA/'before-dense-grid-state.json',s)
    screen=[];later=[[],[],[]];replaced=[]
    for node in s['screen']:
        if node['kind']!='visual_seed':screen.append(node);continue
        original=read(DATA/node['file']);b=original[0]['bbox_source_xyxy'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;frame=original[0]['frame']
        # Rank the first class bundle by the visual proposal's rough dimensions;
        # all other classes remain queued rather than being ruled out.
        candidate_number=int(node['id'].split('-')[1])
        proposal=read(ROOT/'data/drone/mined/object-presence-pilot.json')['candidates'][candidate_number]['bbox_source_xyxy']
        vw=proposal[2]-proposal[0];vh=proposal[3]-proposal[1]
        def proximity(label):
            w,h=size(model,label,cy)
            return min(abs(math.log(vw/w))+abs(math.log(vh/h)),abs(math.log(vw/h))+abs(math.log(vh/w)))
        labels=sorted(model,key=proximity);replaced.append({'root':node['id'],'old_file':node['file']})
        for batch in range(4):
            selected=labels[batch*4:(batch+1)*4]
            items=[h for label in selected for h in local_grid(frame,label,cx,cy,model)]
            root_id=node['id'] if batch==0 else node['id']+f'-grid{batch+1}'
            filename=f'groups/dense-{root_id}.json';atomic(DATA/filename,items)
            meta={'id':root_id,'root':root_id,'kind':'local_grid','label':f'Dense candidate grid in frame {frame}: '+', '.join(selected),
                  'file':filename,'hypothesis_count':len(items),'known':False,'stop_after_first':True,'candidate_family':node['id']}
            s['roots'][root_id]={**meta,'status':'pending','seeds_found':0}
            (screen if batch==0 else later[batch-1]).append(meta)
    s['screen']=screen+[x for phase in later for x in phase]
    s['schedule']='seed_first';s['screen_credits']=0
    s['events'].append({'at':now(),'message':'Dense reference-sized local grids installed; isolate existing positive areas first, then alternate with new searches.'})
    atomic(DATA/'state.json',s)
    change={'created_at':now(),'replaced_unqueried_candidates':replaced,'remaining_screen_groups':len(s['screen']),
            'existing_positive_groups_preserved':len(s['work']),'schedule':'One isolated seed, then up to two new group screens',
            'query_cap':s['max_runs'],'queries_already_reserved':s['runs_reserved'],'benchmark':report['summary']}
    atomic(DATA/'dense-grid-migration.json',change);return change

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['benchmark','apply']);a=p.parse_args()
    import json
    report=benchmark() if a.action=='benchmark' else apply()
    print(json.dumps(report.get('summary',report),indent=2))
