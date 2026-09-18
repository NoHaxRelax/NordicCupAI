"""Blind cross-view consensus in native frame coordinates; separate evaluation unit."""
import argparse,copy,collections
from pathlib import Path
from common import read,write
from evaluate_views import iou,metrics
from compare_frontiers import frontier
p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--exclude-clipped-views',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
m=read(a.data/'manifest.json');sources={r['file']:r for r in m['records'] if r['task']=='detector' and r['split']=='dev'};rows=read(a.predictions)['results'];frames={}
if a.exclude_clipped_views:
 excluded={r['file'] for r in read(a.exclude_clipped_views)['records'] if r['split']=='dev' and r['clipped']};rows=[r for r in rows if r['file'] not in excluded]
for row in rows:
 source=sources[row['file']];assert source['frame']==row['frame'] and source['source_region']==row['source_region'] and source['zoom']==row['zoom'];w,h=source['dimensions'];x1,y1,x2,y2=source['source_region'];sx,sy=(x2-x1)/w,(y2-y1)/h;assert abs(sx-sy)<1e-8
 frame=frames.setdefault(row['frame'],dict(frame=row['frame'],truth={},predictions=[]));assert len(source['tracks'])==len(row['truth'])
 def convert(box):return [x1+box[0]*sx,y1+box[1]*sy,x1+box[2]*sx,y1+box[3]*sy]
 for target,track in zip(row['truth'],source['tracks']):
  t=dict(target,bbox=convert(target['bbox']),track=track)
  if track in frame['truth']:
   old=frame['truth'][track];assert t['class']==old['class'];old['bbox']=[min(old['bbox'][0],t['bbox'][0]),min(old['bbox'][1],t['bbox'][1]),max(old['bbox'][2],t['bbox'][2]),max(old['bbox'][3],t['bbox'][3])]
  else:frame['truth'][track]=t
 for pred in row['predictions']:frame['predictions'].append(dict(pred,bbox=convert(pred['bbox']),view=row['file'],zoom=row['zoom']))
# Same-class overlap clusters depend on predictions only, never labels.
clustered=[]
for frame in frames.values():
 remaining=sorted(frame['predictions'],key=lambda p:-p['score']);groups=[]
 while remaining:
  anchor=remaining.pop(0);group=[anchor];rest=[]
  for pred in remaining:
   (group if pred['class']==anchor['class'] and iou(pred['bbox'],anchor['bbox'])>=.5 else rest).append(pred)
  remaining=rest;groups.append(dict(anchor,support=[dict(view=p['view'],zoom=p['zoom'],score=p['score']) for p in group]))
 clustered.append(dict(frame=frame['frame'],truth=list(frame['truth'].values()),predictions=groups))
report={}
for name,views,zooms in [('deduplicated',1,1),('two_views',2,1),('two_zooms',2,2)]:
 summaries=[]
 for threshold in [.01,.05,.1,.25,.5,.75]:
  selected=[]
  for row in clustered:
   preds=[]
   for p in row['predictions']:
    supporters=[s for s in p['support'] if s['score']>=threshold]
    if len({s['view'] for s in supporters})>=views and len({s['zoom'] for s in supporters})>=zooms:preds.append(p)
   selected.append(dict(row,predictions=preds))
  summaries.append(metrics(selected,m['classes'],threshold))
 report[name]=summaries
write(a.output/'clusters.json',dict(results=clustered));write(a.output/'summary.json',dict(complete_views_only=bool(a.exclude_clipped_views),views=len(rows),frames=len(frames),unique_tracks=len({t['track'] for r in clustered for t in r['truth']}),targets=sum(len(r['truth']) for r in clustered),metrics=report,limitations=['Evaluation unit is per-frame physical target, deduplicated across camera views; visible annotation portions are unioned. NOT the earlier 54 overlapping appearances.','Same development flight; view policy and thresholds still need independent validation.','Search predictions use known camera crop geometry, not target locations. Support requires every counted view score to meet the same threshold.','Consensus retains the highest-score box and does not boost its confidence.']))
print({n:[(v['threshold'],v['targets'],v['tp'],v['fp']) for v in values] for n,values in report.items()})
