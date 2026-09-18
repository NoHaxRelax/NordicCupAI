"""Mine classifier crops from reviewed TRAIN views only; preserve all dev records."""
import argparse,copy,math,os,tarfile,time
from collections import Counter
from pathlib import Path
from common import read,write,sha,fingerprint,verify_dataset
from evaluate_views import iou

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--detector',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);a.artifacts.mkdir(exist_ok=True,parents=True)
 import torch
 from ultralytics import YOLO
 from PIL import Image
 torch.set_num_threads(4);m=read(a.source/'manifest.json');verify_dataset(a.source,m);original=copy.deepcopy(m);detector=YOLO(str(a.detector));model_sha=sha(a.detector)
 for row in m['records']:
  for key in ['file','label_file']:
   if key not in row:continue
   dst=a.output/row[key]
   if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);os.link(a.source/row[key],dst)
 additions=[];new_files=[];started=time.monotonic();views=[r for r in m['records'] if r['task']=='detector' and r['split']=='train']
 for index,row in enumerate(views):
  if row['source']=='validation':assert row['frame']<=164
  im=Image.open(a.source/row['file']).convert('RGB');w,h=im.size;truth=[]
  lines=(a.source/row['label_file']).read_text().splitlines();assert len(lines)==len(row['tracks'])==len(row['classes'])
  for j,line in enumerate(lines):
   c,x,y,bw,bh=map(float,line.split());assert m['classes'][int(c)]==row['classes'][j]
   truth.append(dict(label=m['classes'][int(c)],track=row['tracks'][j],bbox=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
  out=detector.predict(str(a.source/row['file']),imgsz=960,conf=.01,iou=.5,device=0,half=True,verbose=False)[0]
  preds=[dict(bbox=b,score=s) for b,s in zip(out.boxes.xyxy.cpu().tolist(),out.boxes.conf.cpu().tolist())]
  def crop_bounds(box):
   x1,y1,x2,y2=box;pad=max(2,.12*max(x2-x1,y2-y1));return [max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(w,math.ceil(x2+pad)),min(h,math.ceil(y2+pad))]
  def save(pred,label,track,overlap,kind,number):
   bounds=crop_bounds(pred['bbox'])
   if bounds[2]-bounds[0]<2 or bounds[3]-bounds[1]<2:return
   if label=='background':assert all(iou(bounds,t['bbox'])==0 for t in truth)
   f=f'classifier/train/{label}/mined-{index:06d}-{kind}{number}.png';dest=a.output/f;dest.parent.mkdir(parents=True,exist_ok=True);crop=im.crop(bounds);crop.save(dest)
   additions.append(dict(task='classifier',split='train',source=row['source'],frame=row['frame'],zoom=row['zoom'],class_name=label,track=track,file=f,sha256=sha(dest),dimensions=list(crop.size),parent_file=row['file'],parent_sha256=row['sha256'],proposal_bbox=pred['bbox'],crop_bounds=bounds,mining_score=pred['score'],matched_iou=overlap))
   new_files.append(f)
  used=set()
  for j,t in enumerate(truth):
   overlap,k=max(((iou(p['bbox'],t['bbox']),k) for k,p in enumerate(preds) if k not in used),default=(0,-1))
   if overlap>=.5:used.add(k);save(preds[k],t['label'],t['track'],overlap,'p',j)
  count=0
  for pred in sorted(preds,key=lambda p:-p['score']):
   if pred['score']<.05 or count>=2:break
   bounds=crop_bounds(pred['bbox'])
   if all(iou(bounds,t['bbox'])==0 for t in truth):save(pred,'background','background',0,'n',count);count+=1
  if index%25==0:write(a.artifacts/'mining-progress.json',dict(views=index+1,total=len(views),added=len(additions),elapsed=time.monotonic()-started))
 m['records']+=additions
 assert [r for r in m['records'] if r['split']=='dev']==[r for r in original['records'] if r['split']=='dev']
 m['config']['experiments']=[dict(id='cls-proposals-resnet',architecture='resnet50',task='classifier',initialization='pretrained',adaptation='full',zoom='mixed',epochs=80,lr=.0001),dict(id='cls-proposals-convnext',architecture='convnext_tiny',task='classifier',initialization='pretrained',adaptation='full',zoom='mixed',epochs=80,lr=.0001)]
 m['derivation']=dict(source_manifest_sha256=sha(a.source/'manifest.json'),detector_sha256=model_sha,positive_iou=.5,negative_confidence=.05,max_negatives_per_view=2,negative_policy='Expanded crop has zero overlap with every reviewed target in complete train view',code_sha256=sha(Path(__file__)))
 m['source_fingerprint']=fingerprint(dict(derivation=m['derivation'],config=m['config']));m['counts']=dict(Counter(r['task']+'/'+r['split'] for r in m['records']));write(a.output/'manifest.json',m);write(a.output/'config.json',m['config']);verify_dataset(a.output,m)
 release=read(a.source/'release.json');release['source_fingerprint']=m['source_fingerprint'];write(a.output/'release.json',release)
 summary=dict(added=len(additions),added_by_class=dict(Counter(r['class_name'] for r in additions)),dev_records_unchanged=True,train_views_only=True,manifest_sha256=sha(a.output/'manifest.json'),source_fingerprint=m['source_fingerprint'],elapsed=time.monotonic()-started)
 with tarfile.open(a.artifacts/'proposal-crops-delta.tar.gz','w:gz') as t:
  for f in new_files+['manifest.json','config.json','release.json']:t.add(a.output/f,arcname=f)
 summary['delta_sha256']=sha(a.artifacts/'proposal-crops-delta.tar.gz');write(a.artifacts/'mining-result.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
