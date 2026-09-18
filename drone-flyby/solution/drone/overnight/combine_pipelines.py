"""Fixed uncalibrated union of direct detections and two-stage recognition."""
import math
from pathlib import Path
from common import read,write
from evaluate_views import metrics,merge_predictions
from compare_frontiers import frontier
root=Path('artifacts/drone-overnight-runpod');out=root/'pipeline-union-v1';out.mkdir(exist_ok=False)
models={'two_stage':read(root/'ensemble-deeper-resnet/arithmetic.json')['results'],'medium_partial':read(root/'medium-partial/evaluation/detector.json')['results'],'original_partial':read(root/'diagnostic-final-partial/evaluation/detector.json')['results']}
base=models['two_stage']
for name,rows in models.items():
 assert len(rows)==len(base)
 for a,b in zip(base,rows):assert all(a[k]==b[k] for k in ['file','zoom','frame','source_region','truth'])
# The geometric-mean score matches the earlier two-stage evaluator convention.
models['two_stage']=[dict(r,predictions=[dict(p,score=math.sqrt(p['score'])) for p in r['predictions']]) for r in base]
variants={'two_stage':['two_stage'],'medium_plus_two_stage':['two_stage','medium_partial'],'original_plus_two_stage':['two_stage','original_partial'],'all_three':['two_stage','medium_partial','original_partial']}
classes=read(Path('data/drone/overnight-runpod-20260918-v2/manifest.json'))['classes'];report={}
for name,members in variants.items():
 rows=[dict(row,predictions=merge_predictions([p for m in members for p in models[m][i]['predictions']])) for i,row in enumerate(base)]
 write(out/(name+'.json'),dict(results=rows));report[name]=dict(frontier=frontier(rows),metrics=[metrics(rows,classes,t) for t in [.01,.05,.1,.25,.5,.75]])
write(out/'summary.json',dict(variants=report,limitations=['Same development views and labels; not a new test. Scores are not calibrated across models. Fixed union uses same-class IoU .5 NMS with no confidence bonus.']))
print({n:dict(max_hits=r['frontier']['points'][-1]['tp'],equal_hits=[(p['tp'],p['fp']) for p in r['frontier']['points'] if p['tp'] in [10,15,18,20,25,30]]) for n,r in report.items()})
