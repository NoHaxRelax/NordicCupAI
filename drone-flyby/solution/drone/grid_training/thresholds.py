"""Exploratory development operating curves; not independent test metrics."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def curve(path):
 rows=json.loads(path.read_text());out=[]
 for threshold in [0,.001,.003,.01,.03,.1,.2,.3,.5,.7,.9]:
  groups=defaultdict(list);neg=[]
  for r in rows:
   score=r['object_score']*max(r['class_scores']);pred=int(np.argmax(r['class_scores']))
   if r['kind']=='background':neg.append(float(score>=threshold));continue
   for c in r['targets']:
    for g in r['class_groups'][str(c)]:groups[c,r['zoom'],g].append(float(pred==c and score>=threshold))
  cz=defaultdict(list)
  for (c,z,g),v in groups.items():cz[c,z].append(float(np.mean(v)))
  recall=float(np.mean([np.mean(v) for v in cz.values()]));fpr=float(np.mean(neg));out.append(dict(threshold=threshold,macro_top1_recall=recall,background_fpr=fpr,score=recall-fpr))
 return dict(curve=out,best_on_development=max(out,key=lambda r:r['score']),note='Threshold chosen on reused small development set. Product of presence and max-class score is an uncalibrated ranking score.')
if __name__=='__main__':
 root=Path('artifacts/drone-grid-training-20260918-v1');result={}
 for p in sorted(root.glob('*/**/predictions.json')):
  if 'smoke' in str(p):continue
  result[str(p.parent.relative_to(root))]=curve(p)
 (root/'threshold-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['best_on_development'] for k,v in result.items()},indent=2))
