"""Separate class discrimination, foreground gating, and background errors."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torchvision import models
from .train import Data,SmallCNN,write

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);data=Data(a.data,1731)
 ck=torch.load(a.run/'best.pt',map_location='cpu',weights_only=False);arch=ck['args']['arch']
 model=SmallCNN() if arch=='small' else getattr(models,arch)(weights=None)
 if arch!='small':model.fc=torch.nn.Linear(model.fc.in_features,17)
 model.load_state_dict(ck['model']);model.cuda().eval();summary={}
 for split in ['train','dev']:
  ids=[i for i,r in enumerate(data.rows) if r['split']==split];rows=[]
  for start in range(0,len(ids),64):
   ix=ids[start:start+64];x,_,_=data.batch(ix,'cuda');scores=model(x).sigmoid().cpu().numpy()
   for i,s in zip(ix,scores):
    r=data.rows[i];truth={t['class_id'] for t in r['annotations'] if t['fully_contained']};pred=int(np.argmax(s[:16]));rows.append(dict(id=r['id'],positive=r['kind']=='positive',zoom=r['zoom'],pred=data.m['classes'][pred],known_classes=[data.m['classes'][c] for c in sorted(truth)],top1_matches_known=pred in truth,object_score=float(s[16]),max_class_score=float(max(s[:16])),true_class_max=float(max((s[c] for c in truth),default=0))))
  pos=[r for r in rows if r['positive']];neg=[r for r in rows if not r['positive']]
  summary[split]=dict(positive_crops=len(pos),negative_crops=len(neg),crop_top1_known_accuracy=float(np.mean([r['top1_matches_known'] for r in pos])),positive_object_recall=float(np.mean([r['object_score']>=.5 for r in pos])),negative_object_false_positive_rate=float(np.mean([r['object_score']>=.5 for r in neg])),positive_object_score_median=float(np.median([r['object_score'] for r in pos])),positive_true_class_score_median=float(np.median([r['true_class_max'] for r in pos])))
  write(a.run/f'diagnostic-{split}-predictions.json',rows)
 write(a.run/'diagnosis.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
