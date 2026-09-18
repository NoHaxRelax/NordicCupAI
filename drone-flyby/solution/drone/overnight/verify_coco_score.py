"""Check saved-view AP with the competition's COCO library and parameters."""
import argparse,json,contextlib,io,importlib.metadata
from pathlib import Path
import numpy as np
from faster_coco_eval import COCO,COCOeval_faster
from score_saved_ap import score

def coco_score(rows):
 classes=sorted({t['class'] for r in rows for t in r['truth']}|{p['class'] for r in rows for p in r['predictions']});ids={c:i+1 for i,c in enumerate(classes)};present=sorted({t['class'] for r in rows for t in r['truth']});annotations=[];preds=[]
 for i,row in enumerate(rows,1):
  for t in row['truth']:
   x,y,x2,y2=t['bbox'];annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=ids[t['class']],bbox=[x,y,x2-x,y2-y],area=(x2-x)*(y2-y),iscrowd=0))
  for p in row['predictions']:
   x,y,x2,y2=p['bbox']
   if x2>x and y2>y:preds.append(dict(image_id=i,category_id=ids[p['class']],bbox=[x,y,x2-x,y2-y],score=p['score']))
 gt=dict(info={},licenses=[],images=[dict(id=i,file_name=r['file'],width=960,height=540) for i,r in enumerate(rows,1)],categories=[dict(id=ids[c],name=c) for c in classes],annotations=annotations)
 if not preds:return dict(map50=0,per_class={c:0 for c in present})
 with contextlib.redirect_stdout(io.StringIO()):
  g=COCO(gt);d=g.loadRes(preds);e=COCOeval_faster(g,d,'bbox');e.params.imgIds=list(range(1,len(rows)+1));e.params.catIds=[ids[c] for c in present];e.params.iouThrs=np.array([.5]);e.evaluate();e.accumulate()
 ap={}
 for i,c in enumerate(present):
  values=e.eval['precision'][0,:,i,0,-1];ap[c]=float(values[values>-1].mean())
 return dict(map50=sum(ap.values())/len(ap),per_class=ap)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('predictions',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args();report={}
 for path in a.predictions:
  rows=json.loads(path.read_text())['results'];official=coco_score(rows);local=score(rows);delta=abs(official['map50']-local['map50']);report[str(path)]=dict(coco=official,local=local,absolute_difference=delta);assert delta<1e-6,(path,delta)
 out=dict(library_version=importlib.metadata.version('faster-coco-eval'),results=report,limitation='Scoring-library parity on saved development views only; not full-flight replay or official evaluation.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
