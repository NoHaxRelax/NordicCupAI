"""Per-view detector ensembles on saved blind predictions and identical labels."""
from pathlib import Path
from common import read,write
from evaluate_views import metrics,iou
root=Path('artifacts/drone-overnight-runpod')
paths={'original_partial':'diagnostic-final-partial','extra_partial':'diagnostic-extra-partial','extra_full':'diagnostic-extra-full'}
models={name:read(root/folder/'evaluation/detector.json')['results'] for name,folder in paths.items()}
base=models['original_partial']
for rows in models.values():
 assert len(rows)==len(base)
 for a,b in zip(base,rows):
  for k in ['file','zoom','frame','source_region','truth']:assert a[k]==b[k]
def merge(groups):
 kept=[]
 for p in sorted([dict(p) for group in groups for p in group],key=lambda p:-p['score']):
  if not any(p['class']==k['class'] and iou(p['bbox'],k['bbox'])>.5 for k in kept):kept.append(p)
 return kept
variants={**models,'union_original_full':[],'union_all':[],'zoom_routed_exploratory':[]}
for i,row in enumerate(base):
 variants['union_original_full'].append(dict(row,predictions=merge([base[i]['predictions'],models['extra_full'][i]['predictions']])))
 variants['union_all'].append(dict(row,predictions=merge([rows[i]['predictions'] for rows in models.values()])))
 name={0:'extra_partial',1:'extra_full',2:'original_partial'}[row['zoom']]
 variants['zoom_routed_exploratory'].append(models[name][i])
classes=read(Path('data/drone/overnight-runpod-20260917-v1/manifest.json'))['classes']
out=root/'detector-ensemble-v1';out.mkdir(exist_ok=False)
for name,rows in variants.items():write(out/(name+'.json'),dict(results=rows))
write(out/'summary.json',dict(metrics={name:{'all':[metrics(rows,classes,t) for t in [.01,.05,.1,.25,.5,.75]],**{f'L{z}':[metrics([r for r in rows if r['zoom']==z],classes,t) for t in [.01,.05,.1,.25,.5,.75]] for z in range(3)}} for name,rows in variants.items()},limitations=['Same162complete dev views,54overlapping target appearances,5physical tracks.','Uncalibrated maximum score union with same-class IoU.5 NMS; no confidence boost for agreement.','Zoom routing is selected from development results and requires independent validation; not a certified flight policy.']))
print({name:[(p['threshold'],p['tp'],p['fp']) for p in r['all']] for name,r in read(out/'summary.json')['metrics'].items()})
