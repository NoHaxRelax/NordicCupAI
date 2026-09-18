"""Conditional four-way expert on fixed blind predictions, scored as 16 classes."""
import argparse,math,time
from pathlib import Path
from common import read,write,sha
from evaluate_views import metrics
from compare_frontiers import frontier
GROUPS=['mine_roller','large_launcher','other','background']
def apply_expert(pred,prob,reject_other=False):
 if pred['class']!='large_launcher':return dict(pred)
 assert len(prob)==4 and all(math.isfinite(v) and 0<=v<=1 for v in prob)
 label=max(range(4),key=prob.__getitem__)
 if label==0 and prob[0]>=.5:return dict(pred,**{'class':'mine_roller','score':pred['score']*prob[0],'original_class':'large_launcher'})
 if reject_other and label in [2,3] and prob[label]>=.5:return None
 return dict(pred)
def main():
 p=argparse.ArgumentParser()
 for n in ['data','checkpoint','predictions','output']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--device',default='0');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 m=read(a.data/'manifest.json');saved=read(a.predictions);assert saved['manifest_sha256']==sha(a.data/'manifest.json')
 import torch
 from PIL import Image,ImageOps
 from torchvision.transforms import functional as F
 from torchvision.models import resnet50,convnext_tiny
 torch.set_num_threads(4);checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert checkpoint['classes']==GROUPS;arch=checkpoint['spec'].get('architecture','resnet50');size=checkpoint['spec'].get('input_size',224)
 if arch=='resnet50':model=resnet50(weights=None);model.fc=torch.nn.Linear(2048,4)
 elif arch=='convnext_tiny':model=convnext_tiny(weights=None);model.classifier[2]=torch.nn.Linear(model.classifier[2].in_features,4)
 else:raise ValueError('Unsupported expert architecture')
 device=torch.device('cpu' if a.device=='cpu' else 'cuda:'+a.device);model.load_state_dict(checkpoint['model']);model.to(device).eval();sources={r['file']:r for r in m['records'] if r['task']=='detector' and r['split']=='dev'};reports={name:[] for name in ['baseline','relabel','relabel_reject']};details=[];start=time.monotonic()
 for row in saved['results']:
  source=sources[row['file']];assert sha(a.data/row['file'])==source['sha256'];im=Image.open(a.data/row['file']).convert('RGB');w,h=im.size;indices=[i for i,pred in enumerate(row['predictions']) if pred['class']=='large_launcher'];probabilities={}
  for offset in range(0,len(indices),64):
   selected=indices[offset:offset+64];batch=[]
   for index in selected:
    x1,y1,x2,y2=row['predictions'][index]['bbox'];pad=max(2,.12*max(x2-x1,y2-y1));crop=im.crop((max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(w,math.ceil(x2+pad)),min(h,math.ceil(y2+pad))));crop=ImageOps.pad(crop,(size,size),method=Image.Resampling.BILINEAR,color=(114,114,114));batch.append(F.normalize(F.to_tensor(crop),[.485,.456,.406],[.229,.224,.225]))
   with torch.inference_mode():probs=model(torch.stack(batch).to(device)).softmax(1).cpu().tolist()
   probabilities.update(zip(selected,probs))
  reports['baseline'].append(row)
  for name,reject in [('relabel',False),('relabel_reject',True)]:
   preds=[]
   for i,pred in enumerate(row['predictions']):
    result=apply_expert(pred,probabilities[i],reject) if i in probabilities else dict(pred)
    if result is not None:preds.append(result)
   reports[name].append(dict(row,predictions=preds))
  details.append(dict(file=row['file'],probabilities=probabilities));write(a.output/'progress.json',dict(views=len(details),total=len(saved['results']),elapsed=time.monotonic()-start))
 for name,rows in reports.items():write(a.output/(name+'.json'),dict(saved,results=rows))
 write(a.output/'expert-probabilities.json',dict(classes=GROUPS,rows=details));write(a.output/'summary.json',dict(checkpoint_sha256=sha(a.checkpoint),predictions_sha256=sha(a.predictions),elapsed=time.monotonic()-start,variants={name:dict(frontier=frontier(rows),metrics=[metrics(rows,m['classes'],t) for t in [.001,.005,.01,.05,.1,.25,.5]]) for name,rows in reports.items()},limitations=['Only baseline-predicted large launchers invoke the expert; no truth-based crop selection.','Class labels and scoring use the original 16-class development truth. Four-way crop accuracy is a different task.','Mine correction requires expert probability >=.5 and reduces, never increases, the original score. Rejection is a separate fixed variant.']))
if __name__=='__main__':main()
