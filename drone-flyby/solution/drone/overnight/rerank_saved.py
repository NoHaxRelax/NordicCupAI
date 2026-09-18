"""Compare classifier filtering and relabelling on fixed blind proposals."""
import argparse,math,time
from pathlib import Path
from common import read,sha,write
from evaluate_views import metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--classifier',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-manifest',type=Path);a=p.parse_args()
 a.output.mkdir(exist_ok=False,parents=True)
 m=read(a.data/'manifest.json');saved=read(a.predictions)
 if a.source_manifest:
  original=read(a.source_manifest);assert saved['manifest_sha256']==sha(a.source_manifest)
  assert [r for r in original['records'] if r['task']=='detector' and r['split']=='dev']==[r for r in m['records'] if r['task']=='detector' and r['split']=='dev']
 else:assert saved['manifest_sha256']==sha(a.data/'manifest.json')
 import torch
 from torchvision.models import resnet50
 from torchvision.transforms import functional as F
 from PIL import Image,ImageOps
 torch.set_num_threads(4)
 checkpoint=torch.load(a.classifier,map_location='cpu',weights_only=False);assert checkpoint['classes']==m['classifier_classes']
 size=int(checkpoint.get('spec',{}).get('input_size',224))
 if checkpoint.get('spec',{}).get('architecture','resnet50') in ['convnext_tiny','convnext_small']:
  from torchvision.models import convnext_tiny,convnext_small
  model={'convnext_tiny':convnext_tiny,'convnext_small':convnext_small}[checkpoint['spec']['architecture']](weights=None);model.classifier[2]=torch.nn.Linear(model.classifier[2].in_features,len(checkpoint['classes']))
 else:
  model=resnet50(weights=None);model.fc=torch.nn.Linear(2048,len(checkpoint['classes']))
 model.load_state_dict(checkpoint['model']);model.eval()
 sources={r['file']:r for r in m['records'] if r['task']=='detector' and r['split']=='dev'}
 reports={k:[] for k in ['baseline','same_class_product','background_filter','relabel']};all_probs=[];start=time.monotonic()
 for row in saved['results']:
  src=sources[row['file']];assert sha(a.data/src['file'])==src['sha256'];assert sha(a.data/src['label_file'])==src['label_sha256']
  im=Image.open(a.data/row['file']).convert('RGB');w,h=im.size;preds=row['predictions'];probs=[]
  for offset in range(0,len(preds),64):
   group=preds[offset:offset+64];batch=[]
   for pred in group:
    x1,y1,x2,y2=pred['bbox'];pad=max(2,.12*max(x2-x1,y2-y1));crop=im.crop((max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(w,math.ceil(x2+pad)),min(h,math.ceil(y2+pad))))
    crop=ImageOps.pad(crop,(size,size),method=Image.Resampling.BILINEAR,color=(114,114,114));batch.append(F.normalize(F.to_tensor(crop),[.485,.456,.406],[.229,.224,.225]))
   with torch.inference_mode():probs.extend(model(torch.stack(batch)).softmax(1).tolist())
  output={k:[] for k in reports};output['baseline']=preds
  for pred,prob in zip(preds,probs):
   if pred['class'] in m['classes']:output['same_class_product'].append(dict(pred,score=pred['score']*prob[m['classes'].index(pred['class'])]))
   if prob[-1]<.5:output['background_filter'].append(pred)
   label=max(range(len(prob)),key=prob.__getitem__)
   if label<len(m['classes']):output['relabel'].append(dict(pred,**{'class':m['classes'][label],'score':pred['score']*prob[label],'original_class':pred['class']}))
  for name in reports:reports[name].append(dict(row,predictions=output[name]))
  all_probs.append(dict(file=row['file'],probabilities=probs))
  write(a.output/'progress.json',dict(views=len(all_probs),total=len(saved['results']),elapsed=time.monotonic()-start))
 for name,rows in reports.items():write(a.output/(name+'.json'),dict(manifest_sha256=sha(a.data/'manifest.json'),results=rows))
 write(a.output/'probabilities.json',dict(classes=m['classifier_classes'],rows=all_probs))
 write(a.output/'summary.json',dict(classifier_sha256=sha(a.classifier),predictions_sha256=sha(a.predictions),elapsed=time.monotonic()-start,metrics={name:{f'L{z}':[metrics([r for r in rows if r['zoom']==z],m['classes'],t) for t in [.001,.005,.01,.05,.1,.25,.5]] for z in range(3)} for name,rows in reports.items()},limitations=m['limitations']))
if __name__=='__main__':main()
