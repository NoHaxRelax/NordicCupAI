"""Rotated training-only feature neighbours on held-out crops; no detector claims."""
import argparse,time
from pathlib import Path
from common import read,write,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--adapted',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 import torch,numpy as np
 from PIL import Image,ImageOps
 from torchvision.models import resnet50
 from torchvision.transforms import functional as F
 torch.set_num_threads(4);torch.manual_seed(170926);m=read(a.data/'manifest.json');classes=m['classifier_classes'];tr=[r for r in m['records'] if r['task']=='classifier' and r['split']=='train'];va=[r for r in m['records'] if r['task']=='classifier' and r['split']=='dev'];images={}
 for r in tr+va:
  assert sha(a.data/r['file'])==r['sha256'];images[r['file']]=ImageOps.pad(Image.open(a.data/r['file']).convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color=(114,114,114))
 assert not {r['sha256'] for r in tr}&{r['sha256'] for r in va}
 angles=list(range(0,360,45));bank_rows=[dict(r,angle=angle) for r in tr for angle in angles]
 def extract(model,rows):
  pieces=[]
  with torch.inference_mode():
   for offset in range(0,len(rows),128):
    batch=[]
    for r in rows[offset:offset+128]:
     im=images[r['file']];angle=r.get('angle',0)
     if angle:im=im.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor=(114,114,114))
     batch.append(F.normalize(F.to_tensor(im),[.485,.456,.406],[.229,.224,.225]))
    with torch.autocast('cuda',dtype=torch.float16):features=model(torch.stack(batch).cuda())
    pieces.append(torch.nn.functional.normalize(features.float(),dim=1))
  return torch.cat(pieces)
 results={};started=time.monotonic()
 for name,path in [('pretrained',a.weights/'resnet50-11ad3fa6.pth'),('adapted',a.adapted)]:
  model=resnet50(weights=None)
  if name=='pretrained':model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
  else:
   saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['classes']==classes;assert saved['spec'].get('architecture','resnet50')=='resnet50';model.fc=torch.nn.Linear(2048,len(classes));model.load_state_dict(saved['model'])
  model.fc=torch.nn.Identity();model.cuda().eval();bank=extract(model,bank_rows);query=extract(model,va);similarities=query@bank.T
  variants={}
  for source in ['mixed','reference']:
   for zoom_match in [False,True]:
    guesses=[];neighbors=[]
    for i,row in enumerate(va):
     indices=[j for j,b in enumerate(bank_rows) if (source=='mixed' or b['source']=='reference') and (not zoom_match or b['zoom']==row['zoom'])]
     assert indices
     values=similarities[i,indices];vals,positions=values.topk(min(3,len(indices)));selected=[indices[j] for j in positions.tolist()];best=bank_rows[selected[0]];guesses.append(best['class_name']);neighbors.append(dict(file=row['file'],truth=row['class_name'],guess=best['class_name'],matches=[dict(file=bank_rows[j]['file'],angle=bank_rows[j]['angle'],label=bank_rows[j]['class_name'],cosine=float(v)) for j,v in zip(selected,vals.tolist())]))
    per_zoom={}
    for z in range(3):
     per={}
     for c in classes:
      inds=[i for i,r in enumerate(va) if r['zoom']==z and r['class_name']==c]
      if inds:per[c]=dict(n=len(inds),recall=sum(guesses[i]==c for i in inds)/len(inds))
     per_zoom[f'L{z}']=dict(foreground_macro_accuracy=float(np.mean([v['recall'] for c,v in per.items() if c!='background'])),per_class=per)
    key=source+('_same_zoom' if zoom_match else '_all_zooms');variants[key]=per_zoom;write(a.output/f'{name}-{key}-neighbors.json',neighbors)
  results[name]=dict(checkpoint_sha256=sha(path),metrics=variants);write(a.output/'progress.json',dict(completed=list(results),elapsed=time.monotonic()-started));del model,bank,query,similarities;torch.cuda.empty_cache()
 write(a.output/'summary.json',dict(results=results,angles=angles,train_crops=len(tr),dev_crops=len(va),manifest_sha256=sha(a.data/'manifest.json'),elapsed=time.monotonic()-started,limitations=m['limitations']+['Training-only nearest neighbours on oracle crops, not blind object detection. Cosine scores are not calibrated probabilities.']))
if __name__=='__main__':main()
