"""Persistent group testing, independent of the network and live service."""
from pathlib import Path
import re
from common import atomic, read, split, digest, iou, W, H

class Search:
    def __init__(self, folder, state):
        if not 1<=state['max_runs']<=100:raise ValueError('Persisted budget exceeds the hard limit')
        self.folder=Path(folder);self.state=state

    @classmethod
    def create(cls,folder,roots,seeds,limit=100):
        if not 1<=limit<=100:raise ValueError('Budget must be 1..100')
        state={'version':1,'max_runs':limit,'runs_reserved':0,'runs_completed':0,
               'screen':[dict(r,root=r['id'],known=False) for r in roots],'work':[],
               'roots':{r['id']:{**r,'status':'pending','seeds_found':0} for r in roots},
               'seeds':seeds,'initial_seed_count':len(seeds),'pending':None,'events':[]}
        obj=cls(folder,state);obj.save();return obj

    def save(self):atomic(self.folder/'state.json',self.state)
    def items(self,node):return read(self.folder/node['file'])

    def next_node(self):
        s=self.state
        if s['pending']:return s['pending']['node']
        if s['runs_reserved']>=s['max_runs']:return None
        if s.get('retries'):return s['retries'].pop(0)
        seed_first=s.get('schedule')=='seed_first'
        if s['screen'] and (not seed_first or not s['work'] or s.get('screen_credits',0)>0):
            if seed_first:s['screen_credits']=max(0,s.get('screen_credits',0)-1)
            return s['screen'].pop(0)
        if seed_first and s['work']:
            focus=s.get('focus_root')
            if not any(x['root']==focus for x in s['work']):
                chosen=min(s['work'],key=lambda x:(s['roots'][x['root']]['seeds_found'],
                    s['roots'][x['root']]['kind'] not in ('visual_seed','local_grid')))
                focus=chosen['root'];s['focus_root']=focus
            own=[x for x in s['work'] if x['root']==focus]
            s['work']=own+[x for x in s['work'] if x['root']!=focus]
        while s['work']:
            node=s['work'].pop(0);items=self.items(node)
            if not node['known'] or len(items)==1:return node
            left,right=split(items)
            children=[]
            for suffix,values in [('a',left),('b',right)]:
                child={**node,'id':node['id']+'-'+suffix,'known':False,'hypothesis_count':len(values)}
                child.pop('infer_sibling_on_zero',None)
                child['file']='nodes/'+child['id']+'.json'
                atomic(self.folder/child['file'],values);children.append(child)
            children[0]['infer_sibling_on_zero']=children[1]['id']
            s['work'][0:0]=children
        return None

    def reserve(self,node):
        s=self.state
        if s['pending']:raise RuntimeError('Unresolved prior query')
        if s['runs_reserved']>=s['max_runs']:raise RuntimeError('Query cap reached')
        s['runs_reserved']+=1
        q={'query_id':f'q{s["runs_reserved"]:03d}','node':node,'phase':'prepared'}
        s['pending']=q;self.save();return q

    def recover_incomplete(self,score,missing,receipt,reason,clean_zero=False):
        """A finished attempt can be retried, but is never silently called absent.

        With a clean zero score, received frames have already tested negative,
        so only missing frames need another query. Otherwise retry the full
        group. Preserve sibling inference until the combined test is complete.
        Every retry is a new budget reservation, never a repeat of an unknown POST.
        """
        s=self.state;p=s['pending'];node=p['node'];items=self.items(node)
        missing=set(missing);narrow=clean_zero and score==0 and bool(missing)
        values=[x for x in items if x['frame'] in missing] if narrow else items
        if not values:raise RuntimeError('No hypotheses in recovery plan')
        attempt=node.get('recovery_attempts',0)+1
        recovered={**node,'id':node['id']+'-retry-'+p['query_id'],
                   'recovery_attempts':attempt,'hypothesis_count':len(values)}
        recovered['file']='nodes/'+recovered['id']+'.json'
        recovered['label']=s['roots'][node['root']]['label']+(' (missing-frame retry)' if narrow else ' (transport retry)')
        recovered['coverage_receipts']=node.get('coverage_receipts',[])+[receipt]
        atomic(self.folder/recovered['file'],values)
        decision={'query_id':p['query_id'],'reason':reason,'missing_frames':sorted(missing),
                  'scope':'missing_frames' if narrow else 'full_group','attempt':attempt,
                  'retry_hypotheses':len(values),'receipt':receipt,'node':recovered}
        if attempt<=3:
            s.setdefault('retries',[]).insert(0,recovered);decision['action']='retry'
        else:
            s.setdefault('unresolved',[]).append(recovered);decision['action']='deferred'
            root=s['roots'][node['root']]
            if root['status']=='pending':root['status']='incomplete'
        atomic(self.folder/'queries'/p['query_id']/'disposition.json',decision)
        s.setdefault('incomplete_attempts',[]).append({k:v for k,v in decision.items() if k!='node'})
        s['runs_completed']+=1;s['pending']=None;self.save()
        return decision

    def observe(self,score,receipt):
        s=self.state;p=s['pending'];node=p['node'];items=self.items(node)
        root=s['roots'][node['root']]
        if score>0:
            if len(items)==1:
                h=items[0]
                duplicate=any(a['frame']==h['frame'] and a['class']==h['class'] and iou(a['bbox_source_xyxy'],h['bbox_source_xyxy'])>=.5 for a in s['seeds'])
                if not duplicate:
                    b=h['bbox_source_xyxy']
                    s['seeds'].append({**h,'seed_id':p['query_id'],'evidence':'score_confirmed',
                        'bbox_normalized_xyxy':[b[0]/W,b[1]/H,b[2]/W,b[3]/H],
                        'score':score,'receipt':receipt,'query_id':p['query_id'],
                        'box_certainty':'IoU >= 0.50; not exact organizer ground truth'})
                    root['seeds_found']+=1
                root['status']='seed_found'
                if root['stop_after_first']:
                    s['work']=[x for x in s['work'] if x['root']!=node['root']]
                    family=root.get('candidate_family')
                    if family:
                        related={k for k,r in s['roots'].items() if r.get('candidate_family')==family and k!=node['root']}
                        for k in related:s['roots'][k]['status']='skipped_after_family_seed'
                        s['screen']=[x for x in s['screen'] if x['root'] not in related]
                        s['work']=[x for x in s['work'] if x['root'] not in related]
                else:
                    # Give other positive roots their first isolated anchor.
                    own=[x for x in s['work'] if x['root']==node['root']]
                    s['work']=[x for x in s['work'] if x['root']!=node['root']]+own
                if s.get('schedule')=='seed_first':
                    s['focus_root']=None;s['screen_credits']=2
            else:
                node['known']=True
                if root['status']=='pending':
                    root['status']='positive_group';s['work'].append(node)
                else:s['work'].insert(0,node)
        else:
            if node['known']:
                raise RuntimeError('A previously positive group became negative; stop for consistency review')
            if root['status']=='pending':root['status']='no_match_in_tested_hypotheses'
            sibling=node.get('infer_sibling_on_zero')
            if sibling:
                found=False
                for x in s['work']:
                    if x['id']==sibling:x['known']=True;found=True;break
                if not found:raise RuntimeError('Missing sibling in search journal')
        s['runs_completed']+=1;s['pending']=None;self.save()


def missing_frames(plan,receipts):
    frames=plan['predictions_by_frame'];expected=set(map(int,frames))
    received={r['frame'] for r in receipts if r.get('plan_hash')==plan['plan_hash'] and
              r.get('annotation_count')==len(frames.get(str(r['frame']),[]))}
    return sorted(expected-received)


def coverage_ok(plan,receipts,result):
    """Transport receipts plus a clean evaluator result are required even for zero."""
    if relevant_errors(plan,result):return False,'Evaluator reported errors affecting tested frames or an unknown scope'
    missing=missing_frames(plan,receipts)
    if missing:return False,'Missing tested frames: '+','.join(map(str,missing[:12]))
    return True,None


def relevant_errors(plan,result):
    """Failures on frames with no predictions cannot invalidate tested-frame evidence.

    Unrecognized errors remain blocking. Exact response receipts are still
    required for every targeted frame, even when unrelated failures are ignored.
    """
    targets=set(map(int,plan['predictions_by_frame']));relevant=[]
    for error in result.get('errors',[]):
        match=re.search(r'\b(?:HTTP error for frame|Frame)\s+(\d+)\b',error,re.I) if isinstance(error,str) else None
        if not match or int(match.group(1)) in targets:relevant.append(error)
    return relevant
