"""101-point AP50 diagnostic on saved views, not a real-time full-flight score."""
import argparse,json
from pathlib import Path
from evaluate_views import iou

def score(rows):
 classes=sorted({t['class'] for r in rows for t in r['truth']});aps={}
 for c in classes:
  truth={i:[t for t in r['truth'] if t['class']==c] for i,r in enumerate(rows)}
  total=sum(map(len,truth.values()));used={i:set() for i in truth}
  preds=sorted([(p['score'],i,p) for i,r in enumerate(rows) for p in sorted([p for p in r['predictions'] if p['class']==c],key=lambda p:-p['score'])[:100]],key=lambda x:-x[0])
  tp=fp=0;points=[]
  for _,i,p in preds:
   ov,j=max([(iou(p['bbox'],t['bbox']),j) for j,t in enumerate(truth[i]) if j not in used[i]],default=(0,-1))
   if ov>=.5:tp+=1;used[i].add(j)
   else:fp+=1
   points.append((tp/total,tp/(tp+fp)))
  aps[c]=sum(max((pr for re,pr in points if re>=k/100),default=0) for k in range(101))/101
 return dict(map50=sum(aps.values())/len(aps) if aps else None,per_class=aps)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('predictions',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=dict(protocol='101-point interpolated AP50; max100 detections per view per class; represented classes only. Overlapping development views, not full-flight competition replay.',results={str(p):score(json.loads(p.read_text())['results']) for p in a.predictions});a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
