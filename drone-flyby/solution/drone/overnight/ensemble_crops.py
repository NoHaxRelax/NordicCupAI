"""Fixed-checkpoint crop ensemble diagnostic; not blind detection."""
import argparse
from pathlib import Path
from common import read,sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 import torch,numpy as np
 from PIL import Image,ImageOps
 from torchvision.models import resnet50
 from torchvision.transforms import functional as F
 torch.set_num_threads(4);m=read(a.data/'manifest.json');rows=[r for r in m['records'] if r['task']=='classifier' and r['split']=='dev'];classes=m['classifier_classes'];xs=[]
 for r in rows:
  assert sha(a.data/r['file'])==r['sha256'];im=Image.open(a.data/r['file']).convert('RGB');im=ImageOps.pad(im,(224,224),method=Image.Resampling.BILINEAR,color=(114,114,114));xs.append(F.normalize(F.to_tensor(im),[.485,.456,.406],[.229,.224,.225]))
 paths={'mixed_original':a.run_root/'drone-night-local-cache-v3/experiments/cls-full','mixed_extra':a.run_root/'drone-night-cls-extra-v3/experiments/cls-full-extra-views',**{f'L{z}':a.run_root/f'drone-night-cls-zoom-v1/experiments/cls-L{z}-extra' for z in range(3)}}
 probabilities={};hashes={}
 for name,path in paths.items():
  result=read(path/'result.json');assert sha(path/'best.pt')==result['checkpoint_sha256'];hashes[name]=result['checkpoint_sha256'];saved=torch.load(path/'best.pt',map_location='cpu',weights_only=False);assert saved['classes']==classes
  model=resnet50(weights=None);model.fc=torch.nn.Linear(2048,len(classes));model.load_state_dict(saved['model']);model.eval();ps=[]
  with torch.inference_mode():
   for i in range(0,len(xs),32):ps.extend(model(torch.stack(xs[i:i+32])).softmax(1).tolist())
  probabilities[name]=np.array(ps);del model,saved
  write(a.output/'progress.json',dict(completed_models=list(probabilities)))
 mixed=(probabilities['mixed_original']+probabilities['mixed_extra'])/2
 routed=np.stack([probabilities[f'L{r["zoom"]}'][i] for i,r in enumerate(rows)])
 probabilities.update(mixed_pair=mixed,zoom_routed=routed,mixed_plus_specialist=(mixed+routed)/2,all_five=sum(probabilities.values())/5)
 def evaluate(ps):
  guesses=ps.argmax(1);report={}
  for z in range(3):
   subset=[i for i,r in enumerate(rows) if r['zoom']==z];per={}
   for c in classes:
    inds=[i for i in subset if rows[i]['class_name']==c]
    if inds:per[c]=dict(n=len(inds),recall=sum(classes[int(guesses[i])]==c for i in inds)/len(inds))
   report[f'L{z}']=dict(foreground_macro_accuracy=float(np.mean([v['recall'] for c,v in per.items() if c!='background'])),per_class=per)
  return report
 write(a.output/'probabilities.json',dict(classes=classes,records=[dict(file=r['file'],zoom=r['zoom'],truth=r['class_name']) for r in rows],models={k:v.tolist() for k,v in probabilities.items()}))
 write(a.output/'summary.json',dict(checkpoint_hashes=hashes,metrics={k:evaluate(v) for k,v in probabilities.items()},limitations=m['limitations']+['Oracle crop ensemble diagnostic; not blind detection. Equal weights are exploratory development choices.']))
if __name__=='__main__':main()
