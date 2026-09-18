import argparse,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from .data import Scenes,sha
from .model import AssetNet,loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1500);p.add_argument('--calibrated-sampling',type=float,default=0.);p.add_argument('--domain-randomization',action='store_true');p.add_argument('--real-per-batch',type=int,default=4);a=p.parse_args()
 if a.steps<1 or not 0<=a.real_per_batch<=8:raise ValueError('Invalid step count or real batch fraction')
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(1824);torch.cuda.manual_seed_all(1824)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 a.output.mkdir(parents=True,exist_ok=False);dataset=Scenes(a.bank,a.data,calibrated_sampling=a.calibrated_sampling,domain_randomization=a.domain_randomization);model=AssetNet(len(dataset.classes)).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps,eta_min=.00005);scaler=torch.amp.GradScaler('cuda')
 receipt=dict(initialization='random',parameters=sum(p.numel()for p in model.parameters()),steps=a.steps,batch=8,patch_size=384,calibrated_sampling=a.calibrated_sampling,domain_randomization=a.domain_randomization,real_per_batch=a.real_per_batch,bank_sha256=sha(a.bank/'manifest.json'),data_sha256=sha(a.data/'manifest.json'),classes=dataset.classes,code_hashes={str(p.name):sha(p)for p in Path(__file__).parent.glob('*.py')},gpu=torch.cuda.get_device_name())
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2));started=time.time()
 for step in range(a.steps):
  batch=[dataset.sample(real=i<a.real_per_batch)for i in range(8)];x,h,b,v=[torch.from_numpy(np.stack([r[k]for r in batch])).cuda()for k in range(4)]
  optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):total,heat,box=loss(model(x),h,b,v)
  if not torch.isfinite(total):raise RuntimeError('Nonfinite loss')
  scaler.scale(total).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10);scaler.step(optimizer);scaler.update();schedule.step()
  if (step+1)%25==0:
   row=dict(step=step+1,steps=a.steps,seconds=time.time()-started,loss=float(total),heat_loss=float(heat),box_loss=float(box),vram_gib=torch.cuda.max_memory_allocated()/2**30);(a.output/'progress.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
  if (step+1)%500==0 or step+1==a.steps:
   torch.save(dict(state_dict={k:v.detach().cpu()for k,v in model.state_dict().items()},classes=dataset.classes,step=step+1,receipt=receipt),a.output/f'step-{step+1}.pt')
 (a.output/'complete.json').write_text(json.dumps(dict(seconds=time.time()-started,weights=f'step-{a.steps}.pt',sha256=sha(a.output/f'step-{a.steps}.pt'))))


if __name__=='__main__':main()
