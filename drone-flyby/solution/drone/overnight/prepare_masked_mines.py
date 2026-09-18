"""Training-only mine-roller background augmentation using deterministic masks."""
import json,hashlib,tarfile,random,copy
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2];source=ROOT/'data/drone/overnight-runpod-20260918-v2';out=ROOT/'artifacts/drone-overnight-runpod/masked-mine-delta';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((source/'manifest.json').read_text());targets=[r for r in m['records'] if r['task']=='classifier' and r['split']=='train' and r['class_name']=='mine_roller' and r['zoom']==2];backgrounds=[r for r in m['records'] if r['task']=='classifier' and r['split']=='train' and r['class_name']=='background' and r['zoom']==2];assert len(targets)==4 and backgrounds
rng=random.Random(170926);records=[];audit=[];files=[]
for i,row in enumerate(targets):
 assert sha(source/row['file'])==row['sha256'];assert row['source']=='reference' or row['frame']<=164
 im=cv2.imread(str(source/row['file']));h,w=im.shape[:2];mask=np.zeros((h,w),np.uint8);cv2.setRNGSeed(0);cv2.grabCut(im,mask,(1,1,w-2,h-2),np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_RECT);mask=np.isin(mask,[1,3]).astype(np.uint8)*255;fraction=float(np.mean(mask>0));assert .04<=fraction<=.95
 mask=cv2.dilate(mask,np.ones((5,5),np.uint8));alpha=cv2.GaussianBlur(mask.astype(np.float32)/255,(3,3),.5)[:,:,None]
 maskfile=f'mask-{i}.png';cv2.imwrite(str(out/maskfile),mask);files.append(maskfile);audit.append(dict(parent=row['file'],mask=maskfile,fraction=fraction))
 for variant in range(24):
  if variant%2==0:
   value=rng.randint(65,180);bg=np.full_like(im,value);bgmeta={'synthetic_gray':value}
  else:
   background=rng.choice(backgrounds);assert sha(source/background['file'])==background['sha256'];bg=cv2.imread(str(source/background['file']));bg=cv2.resize(bg,(w,h));bgmeta={'file':background['file'],'sha256':background['sha256'],'split':'train'}
  rgb=cv2.cvtColor(np.clip(alpha*im+(1-alpha)*bg,0,255).astype(np.uint8),cv2.COLOR_BGR2RGB)
  for zoom in [0,1,2]:
   image=Image.fromarray(rgb);scale=2**(2-zoom);size=(max(2,round(w/scale)),max(2,round(h/scale)));image=image.resize(size,Image.Resampling.LANCZOS);file=f'classifier/train/mine_roller/masked-{i}-{variant:02d}-L{zoom}.png';p=out/file;p.parent.mkdir(parents=True,exist_ok=True);image.save(p);files.append(file)
   r=copy.deepcopy(row);r.update(file=file,sha256=sha(p),zoom=zoom,dimensions=list(size),augmentation={'kind':'deterministic_foreground_background_swap','parent_file':row['file'],'parent_sha256':row['sha256'],'mask_sha256':sha(out/maskfile),'background':bgmeta});records.append(r)
receipt=dict(source_manifest_sha256=sha(source/'manifest.json'),records=records,audit=audit,code_sha256=sha(Path(__file__)),mask_method='GrabCut3 rectangle border1; fraction .04..95; dilation5; Gaussian feather .5; 12 gray and12 reviewed TRAIN backgrounds per native view, resized to three zooms',inspiration_sha256=sha(ROOT/'drone/template_matching/build.py'))
(out/'augmentation.json').write_text(json.dumps(receipt,indent=2));files.append('augmentation.json')
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:
 for f in files:t.add(out/f,arcname=f)
print(json.dumps({'added':len(records),'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}))
