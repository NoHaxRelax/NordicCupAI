"""Paired crop recognition training. No API calls or detector claims."""
import argparse, collections, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from PIL import Image


def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class SmallCNN(nn.Module):
 def __init__(self):
  super().__init__();layers=[];c=3
  for out,stride in [(32,1),(32,2),(64,1),(64,2),(128,1),(128,2)]:
   layers.extend([nn.Conv2d(c,out,3,stride,1,bias=False),nn.BatchNorm2d(out),nn.SiLU()]);c=out
  self.features=nn.Sequential(*layers);self.head=nn.Linear(c,17)
 def forward(self,x):return self.head(F.adaptive_avg_pool2d(self.features(x),1).flatten(1))

class Data:
 def __init__(self,path,seed):
  self.path=Path(path);self.m=json.loads((self.path/'manifest.json').read_text());self.rows=self.m['records'];self.rng=random.Random(seed);self.cache={};self.groups={c:{} for c in self.m['classes']};self.bg={}
  for i,r in enumerate(self.rows):
   assert sha(self.path/r['file'])==r['sha256']
   self.cache[i]=torch.from_numpy(np.array(Image.open(self.path/r['file']).convert('RGB'))).permute(2,0,1)
   if r['split']!='train':continue
   if r['kind']=='background':self.bg.setdefault(r['source'],{}).setdefault(r.get('scene_type','unspecified'),[]).append(i)
   for a in r['annotations']:
    if a['fully_contained']:self.groups[a['class_name']].setdefault(a['group'],{}).setdefault(r['frame'],{}).setdefault(r['zoom'],[]).append(i)
  self.dev=[i for i,r in enumerate(self.rows) if r['split']=='dev']
  assert all(self.groups.values())
 def choose(self,d):return self.rng.choice(list(d.values())) if isinstance(d,dict) else self.rng.choice(d)
 def sample(self,n):
  ids=[]
  for j in range(n):
   if j<n//2:ids.append(self.choose(self.choose(self.choose(self.choose(self.choose(self.groups))))))
   else:ids.append(self.choose(self.choose(self.choose(self.bg))))
  self.rng.shuffle(ids);return ids
 def batch(self,ids,device,augment=False):
  images=torch.stack([self.cache[i] for i in ids]).to(device).float()/255
  if augment:
   # Right-angle rotations/flips preserve all labelled pixels. Colour changes do
   # not introduce fabricated background or crop a known object out of the image.
   images=torch.stack([torch.rot90(x,self.rng.randrange(4),[-2,-1]).flip(-1) if self.rng.random()<.5 else torch.rot90(x,self.rng.randrange(4),[-2,-1]) for x in images])
   brightness=torch.empty(len(ids),1,1,1,device=device).uniform_(.85,1.15);contrast=torch.empty_like(brightness).uniform_(.85,1.15)
   mean=images.mean((-2,-1),keepdim=True);images=((images-mean)*contrast+mean)*brightness;images=images.clamp(0,1)
  targets=torch.zeros(len(ids),17,device=device);mask=torch.zeros_like(targets)
  for j,i in enumerate(ids):
   r=self.rows[i];mask[j,:16]=float(r['annotation_complete']);mask[j,16]=1;targets[j,16]=float(r['kind']=='positive')
   for a in r['annotations']:targets[j,a['class_id']]=1;mask[j,a['class_id']]=1
  images=(images-images.new_tensor([.485,.456,.406])[None,:,None,None])/images.new_tensor([.229,.224,.225])[None,:,None,None]
  return images,targets,mask

def loss_fn(logits,y,mask):
 raw=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
 # Equal per-sample weight irrespective of number of known classes.
 cls=(raw[:,:16]*mask[:,:16]).sum(1)/mask[:,:16].sum(1).clamp_min(1)
 return cls.mean()+raw[:,16].mean()

@torch.no_grad()
def evaluate(model,data,device,batch):
 model.eval();preds=[]
 for start in range(0,len(data.dev),batch):
  ids=data.dev[start:start+batch];x,y,m=data.batch(ids,device);p=model(x).sigmoid().cpu().numpy()
  for idx,prob in zip(ids,p):
   r=data.rows[idx];preds.append(dict(id=r['id'],file=r['file'],zoom=r['zoom'],scene_type=r.get('scene_type','unspecified'),kind=r['kind'],targets=sorted({a['class_id'] for a in r['annotations'] if a['fully_contained']}),groups=sorted({a['group'] for a in r['annotations'] if a['fully_contained']}),class_groups={str(c):sorted({a['group'] for a in r['annotations'] if a['fully_contained'] and a['class_id']==c}) for c in {a['class_id'] for a in r['annotations'] if a['fully_contained']}},class_scores=prob[:16].tolist(),object_score=float(prob[16])))
 metrics={};track=collections.defaultdict(list);top1=collections.defaultdict(list);neg=[];errors=[]
 for r in preds:
  if r['kind']=='background':
   false=r['object_score']>=.5 and max(r['class_scores'])>=.5;neg.append(float(false))
   if false:errors.append(dict(r,error='background_false_positive'))
  else:
   for c in r['targets']:
    ok=r['object_score']>=.5 and r['class_scores'][c]>=.5
    for g in r['class_groups'][str(c)]:
     track[c,r['zoom'],g].append(float(ok))
     top1[c,r['zoom'],g].append(float(ok and int(np.argmax(r['class_scores']))==c))
    if not ok or int(np.argmax(r['class_scores']))!=c:errors.append(dict(r,error='missed_or_confused_known_class',missed_class=c))
 for c in range(16):
  for z in range(3):
   vals=[np.mean(v) for (cc,zz,g),v in track.items() if cc==c and zz==z]
   if vals:metrics[data.m['classes'][c]+f'/L{z}']=float(np.mean(vals))
 macro=float(np.mean(list(metrics.values()))) if metrics else 0
 topmetrics={}
 for c in range(16):
  for z in range(3):
   vals=[np.mean(v) for (cc,zz,g),v in top1.items() if cc==c and zz==z]
   if vals:topmetrics[data.m['classes'][c]+f'/L{z}']=float(np.mean(vals))
 topmacro=float(np.mean(list(topmetrics.values()))) if topmetrics else 0
 fpr=float(np.mean(neg)) if neg else None
 byscene={}
 for scene in sorted({r['scene_type'] for r in preds if r['kind']=='background'}):
  vals=[float(r['object_score']>=.5 and max(r['class_scores'])>=.5) for r in preds if r['kind']=='background' and r['scene_type']==scene];byscene[scene]=float(np.mean(vals))
 return dict(macro_known_positive_recall=macro,background_false_positive_rate=fpr,selection_score=topmacro-(fpr or 0),macro_known_class_top1=topmacro,per_class_zoom=metrics,top1_per_class_zoom=topmetrics,background_by_scene=byscene,threshold=.5,limitations='Development only, partial positives, seven classes; repeated views are correlated. Not mAP or flight score.'),preds,errors

def run(a):
 out=Path(a.output or os.environ.get('NORDIC_RUN_DIR',''));out.mkdir(parents=True,exist_ok=True)
 if (out/'started.json').exists():raise ValueError('Refusing to overwrite experiment')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
 if not torch.cuda.is_available() and not a.cpu:raise RuntimeError('GPU unavailable')
 device='cpu' if a.cpu else 'cuda';data=Data(a.data,a.seed)
 if a.arch=='small':model=SmallCNN()
 else:
  ctor=getattr(models,a.arch);weights=getattr(models,'ResNet18_Weights' if a.arch=='resnet18' else 'ResNet34_Weights').DEFAULT if a.pretrained else None
  model=ctor(weights=weights);model.fc=nn.Linear(model.fc.in_features,17)
 model.to(device);head=model.head if a.arch=='small' else model.fc
 headids={id(p) for p in head.parameters()};backbone=[p for p in model.parameters() if id(p) not in headids]
 optim=torch.optim.AdamW([{'params':backbone,'lr':a.lr*(.1 if a.pretrained else 1)},{'params':head.parameters(),'lr':a.lr}],weight_decay=.01)
 scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');best=-float('inf');stale=0;history=[]
 write(out/'started.json',dict(args=vars(a),manifest_sha256=sha(Path(a.data)/'manifest.json'),torch=torch.__version__,device=str(torch.cuda.get_device_name(0)) if device=='cuda' else 'cpu',parameters=sum(p.numel() for p in model.parameters()),started=time.time()))
 for epoch in range(a.epochs):
  started=time.time();model.train();frozen=a.pretrained and epoch<2
  for p in backbone:p.requires_grad_(not frozen)
  if frozen:
   for module in model.modules():
    if isinstance(module,nn.BatchNorm2d):module.eval()
  losses=[]
  for step in range(a.steps):
   ids=data.sample(a.batch);x,y,mask=data.batch(ids,device,augment=True);optim.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device,enabled=device=='cuda'):loss=loss_fn(model(x),y,mask)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   scaler.scale(loss).backward();scaler.unscale_(optim);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optim);scaler.update();losses.append(float(loss.detach()))
  metrics,preds,errors=evaluate(model,data,device,a.batch);row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-started,**metrics);history.append(row);write(out/'history.json',history)
  write(out/'progress.json',row);print(json.dumps(row),flush=True)
  checkpoint=dict(model=model.state_dict(),args=vars(a),epoch=epoch+1,classes=data.m['classes'],manifest_sha256=sha(Path(a.data)/'manifest.json'))
  torch.save(checkpoint,out/'last.pt')
  if metrics['selection_score']>best+1e-4:
   best=metrics['selection_score'];stale=0;torch.save(checkpoint,out/'best.pt');write(out/'best-metrics.json',row);write(out/'predictions.json',preds);write(out/'errors.json',errors)
  else:stale+=1
  if epoch>=7 and stale>=a.patience:break
 write(out/'complete.json',dict(epochs=len(history),best_selection_score=best,checkpoint_sha256=sha(out/'best.pt'),completed=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output');p.add_argument('--arch',choices=['small','resnet18','resnet34'],default='resnet18');p.add_argument('--pretrained',action='store_true');p.add_argument('--cpu',action='store_true');p.add_argument('--seed',type=int,default=1731);p.add_argument('--batch',type=int,default=32);p.add_argument('--epochs',type=int,default=24);p.add_argument('--steps',type=int,default=50);p.add_argument('--lr',type=float,default=.001);p.add_argument('--patience',type=int,default=6);run(p.parse_args())
