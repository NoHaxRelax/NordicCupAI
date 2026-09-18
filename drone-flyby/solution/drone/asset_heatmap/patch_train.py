"""Train an asset crop verifier with independent foreground/background variation."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from .data import Scenes,sha
from drone.scratch_objects.patch_cnn import PatchCNN,normalize_crop


def foreground_box(mask,rng):
 """Match proposal crop scale while varying loose/tight localization."""
 yy,xx=np.where(mask>32)
 if not len(xx):raise ValueError('Empty foreground mask')
 x1,y1,x2,y2=float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)
 w,h=x2-x1,y2-y1
 # Most proposals surround the object closely; retain some broad proposals.
 margin=float(rng.uniform(0,.2) if rng.random()<.8 else rng.uniform(.2,.5))
 dx,dy=rng.uniform(-.04,.04,2)*[w,h]
 return [max(0,x1-margin*w+dx),max(0,y1-margin*h+dy),
         min(mask.shape[1],x2+margin*w+dx),min(mask.shape[0],y2+margin*h+dy)]


def patch(dataset,label,proposal_crops=False,wide_negatives=False):
 rng=dataset.rng
 bg=dataset.backgrounds[int(rng.integers(len(dataset.backgrounds)))];h,w=bg.shape[:2]
 size=int(rng.integers(35,200));bw=bh=size
 if wide_negatives and not label:
  size=float(np.exp(rng.uniform(np.log(8),np.log(240))));aspect=float(np.exp(rng.uniform(-.7,.7)));bw=min(w,max(4,round(size*aspect)));bh=min(h,max(4,round(size/aspect)))
 x=int(rng.integers(max(1,w-bw+1)));y=int(rng.integers(max(1,h-bh+1)));background_crop=bg[y:y+bh,x:x+bw];image=cv2.resize(background_crop,(64,64))
 if label:
  c=label-1;pool=dataset.calibrated[c]if dataset.calibrated[c]and rng.random()<.6 else dataset.templates[c];source,mask=pool[int(rng.integers(len(pool)))]
  if rng.random()<.2:
   image=normalize_crop(source,foreground_box(mask,rng) if proposal_crops else [0,0,source.shape[1],source.shape[0]])
  else:
   yy,xx=np.where(mask>0);source=source[yy.min():yy.max()+1,xx.min():xx.max()+1];mask=mask[yy.min():yy.max()+1,xx.min():xx.max()+1];h,w=source.shape[:2];s=float(rng.uniform(25,49))/max(h,w);source=cv2.resize(source,None,fx=s,fy=s);mask=cv2.resize(mask,source.shape[1::-1]);h,w=source.shape[:2]
   canvas=np.zeros((80,80,3),np.uint8);alpha=np.zeros((80,80),np.uint8);canvas[(80-h)//2:(80-h)//2+h,(80-w)//2:(80-w)//2+w]=source;alpha[(80-h)//2:(80-h)//2+h,(80-w)//2:(80-w)//2+w]=mask
   mat=cv2.getRotationMatrix2D((39.5,39.5),float(rng.uniform(-180,180)),1.);canvas=cv2.warpAffine(canvas,mat,(80,80));alpha=cv2.warpAffine(alpha,mat,(80,80));canvas=canvas[8:72,8:72];alpha=alpha[8:72,8:72].astype(np.float32)[:,:,None]/255
   color=canvas.astype(np.float32)/255;gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)[:,:,None];color=np.clip(gray+(color-gray)*rng.uniform(.3,1.5),0,1)**rng.uniform(.6,1.6);canvas=np.clip((color*rng.uniform(.45,1.5)+rng.uniform(-.06,.06))*255,0,255)
   image=(canvas*alpha+image*(1-alpha)).astype(np.uint8)
   image=normalize_crop(image,foreground_box(alpha[:,:,0]*255,rng) if proposal_crops else [0,0,64,64])
 else:image=normalize_crop(background_crop,[0,0,bw,bh]) if wide_negatives else normalize_crop(image,[0,0,64,64])
 if rng.random()<.5:image=image[:,::-1].copy()
 image=np.clip(image.astype(np.float32)*rng.uniform(.8,1.2)+rng.uniform(-10,10),0,255)
 return image.transpose(2,0,1).copy()/255


def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=800);p.add_argument('--proposal-crops',action='store_true');p.add_argument('--wide-negatives',action='store_true');p.add_argument('--neutral-tata',action='store_true');a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(1825);torch.cuda.manual_seed_all(1825);dataset=Scenes(a.bank,a.data,seed=1825);classes=['background',*dataset.classes];model=PatchCNN(len(classes)).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0005);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=.0001);a.output.mkdir(parents=True,exist_ok=False)
 if a.neutral_tata:
  index=dataset.classes.index('ta-ta')
  for collection in [dataset.templates,dataset.calibrated]:
   cleaned=[]
   for source,mask in collection[index]:
    neutral=cv2.cvtColor(source,cv2.COLOR_BGR2HSV)[:,:,1]<70;mask=(neutral&(mask>0)).astype(np.uint8)*255
    if np.count_nonzero(mask)>=8:cleaned.append((source,mask))
   collection[index]=cleaned
  if not dataset.templates[index]:raise ValueError('No neutral ta-ta foreground remains')
 receipt=dict(initialization='random',steps=a.steps,batch=128,positive_fraction=.5,calibrated_sampling=.6,proposal_crops=a.proposal_crops,wide_negatives=a.wide_negatives,neutral_tata=a.neutral_tata,classes=classes,bank_sha256=sha(a.bank/'manifest.json'),data_sha256=sha(a.data/'manifest.json'),code_sha256=sha(__file__),gpu=torch.cuda.get_device_name());(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2));start=time.time()
 for step in range(a.steps):
  labels=dataset.rng.integers(1,len(classes),128);labels[:64]=0;dataset.rng.shuffle(labels);x=torch.from_numpy(np.stack([patch(dataset,int(c),a.proposal_crops,a.wide_negatives)for c in labels])).cuda();y=torch.from_numpy(labels).cuda();model.train();optimizer.zero_grad(set_to_none=True);logits=model(x);loss=nn.functional.cross_entropy(logits,y,label_smoothing=.02)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward();optimizer.step();scheduler.step()
  if (step+1)%50==0:
   row=dict(step=step+1,steps=a.steps,seconds=time.time()-start,loss=float(loss.detach()),batch_accuracy=float((logits.argmax(1)==y).float().mean()));(a.output/'progress.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
 torch.save(dict(state_dict={k:v.detach().cpu()for k,v in model.state_dict().items()},classes=classes,seed=1825,receipt=receipt),a.output/'last.pt');(a.output/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,checkpoint_sha256=sha(a.output/'last.pt'))))


if __name__=='__main__':main()
