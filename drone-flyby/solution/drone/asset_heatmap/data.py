"""Online asset rendering, with real context tiles from the approved snapshot."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Scenes:
 def __init__(self,bank,data,seed=1824,calibrated_sampling=0.,domain_randomization=False):
  self.rng=np.random.default_rng(seed);self.bank=Path(bank);self.data=Path(data)
  if not 0<=calibrated_sampling<=1:raise ValueError('Invalid calibration sampling fraction')
  self.calibrated_sampling=calibrated_sampling;self.domain_randomization=domain_randomization
  m=json.loads((self.bank/'manifest.json').read_text());self.classes=sorted(m['classes']);self.templates=[[]for _ in self.classes];self.calibrated=[[]for _ in self.classes]
  for row in m['templates']:
   p=self.bank/row['file'];assert sha(p)==row['sha256'];image=cv2.imread(str(p))
   if row.get('mask_file') and not row.get('mask_fallback'):
    p=self.bank/row['mask_file'];assert sha(p)==row['mask_sha256'];mask=cv2.imread(str(p),0)
   else:
    # A thin valid object can occupy less than the earlier bank's 4% cutoff.
    # Never teach a whole background rectangle as object foreground.
    if row['source']=='reviewed_validation' and row['class']=='large_launcher':continue
    h,w=image.shape[:2];gc=np.zeros((h,w),np.uint8);cv2.setRNGSeed(0)
    cv2.grabCut(image,gc,(1,1,w-2,h-2),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT)
    mask=np.isin(gc,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
    if np.count_nonzero(mask)<8:
     gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);mask=(gray<np.percentile(gray,15)).astype(np.uint8)*255;mask[:2]=0;mask[-2:]=0;mask[:,:2]=0;mask[:,-2:]=0
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
   if row['class']=='small_launcher':
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB);mask=((lab[:,:,1]<126)&(lab[:,:,2]>140)).astype(np.uint8)*255
   if not mask.any():continue
   index=self.classes.index(row['class']);self.templates[index].append((image,mask))
   if row.get('calibration'):self.calibrated[index].append((image,mask))
  assert all(self.templates)
  dataset=json.loads((self.data/'manifest.json').read_text());assert dataset['classes']==self.classes
  self.real=[];self.backgrounds=[]
  for r in dataset['records']:
   if r['split']!='train':continue
   # Native source views only; repetitions need not be loaded more than once.
   if not ('reference_frame'in r or ('reviewed_source'in r and r['file'].endswith('-0.png'))):continue
   p=self.data/r['file'];assert sha(p)==r['sha256'];im=cv2.imread(str(p));h,w=im.shape[:2]
   p=self.data/r['label'];assert sha(p)==r['label_sha256'];boxes=[]
   for line in p.read_text().splitlines():
    c,x,y,bw,bh=map(float,line.split());boxes.append([int(c),(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
   self.real.append((im,boxes))
   if not boxes:self.backgrounds.append(im)
  assert self.backgrounds and self.real

 def sample(self,size=384,real=False):
  rng=self.rng;boxes=[];partial=[]
  if real:
   im,original=self.real[int(rng.integers(len(self.real)))];h,w=im.shape[:2]
   scale=float(rng.uniform(.75,1.3));im=cv2.resize(im,None,fx=scale,fy=scale);original=[[c,*[v*scale for v in b]] for c,*b in original];h,w=im.shape[:2]
   if min(h,w)<size:
    im=cv2.copyMakeBorder(im,0,max(0,size-h),0,max(0,size-w),cv2.BORDER_REFLECT);h,w=im.shape[:2]
   if original and rng.random()<.8:
    c,x1,y1,x2,y2=original[int(rng.integers(len(original)))];x=int(np.clip((x1+x2)/2-rng.uniform(.3,.7)*size,0,w-size));y=int(np.clip((y1+y2)/2-rng.uniform(.3,.7)*size,0,h-size))
   else:x=int(rng.integers(w-size+1));y=int(rng.integers(h-size+1))
   image=im[y:y+size,x:x+size].copy()
   for c,x1,y1,x2,y2 in original:
    if x<=x1<x2<=x+size and y<=y1<y2<=y+size:boxes.append([c,x1-x,y1-y,x2-x,y2-y])
    elif min(x+size,x2)>max(x,x1) and min(y+size,y2)>max(y,y1):partial.append([c,max(0,x1-x),max(0,y1-y),min(size,x2-x),min(size,y2-y)])
  else:
   bg=self.backgrounds[int(rng.integers(len(self.backgrounds)))];h,w=bg.shape[:2];x=int(rng.integers(max(1,w-size+1)));y=int(rng.integers(max(1,h-size+1)));image=bg[y:y+size,x:x+size].copy()
   if image.shape[:2]!=(size,size):image=cv2.resize(image,(size,size))
   for _ in range(int(rng.integers(1,5))):
    c=int(rng.integers(len(self.classes)));choices=self.templates[c]
    if self.calibrated[c] and rng.random()<self.calibrated_sampling:choices=self.calibrated[c]
    crop,mask=choices[int(rng.integers(len(choices)))];h,w=crop.shape[:2]
    factor=float(rng.uniform(.5,1.4));aspect=float(rng.uniform(.7,1.3));w2=max(7,round(w*factor*aspect));h2=max(7,round(h*factor/aspect))
    factor=min(1.,200/max(w2,h2));w2=max(7,round(w2*factor));h2=max(7,round(h2*factor));crop=cv2.resize(crop,(w2,h2));mask=cv2.resize(mask,(w2,h2),interpolation=cv2.INTER_NEAREST)
    mat=cv2.getRotationMatrix2D((w2/2,h2/2),float(rng.uniform(-180,180)),1);corners=cv2.transform(np.float32([[[0,0],[w2,0],[w2,h2],[0,h2]]]),mat)[0];lo=np.floor(corners.min(0));hi=np.ceil(corners.max(0));mat[:,2]-=lo;rw,rh=(hi-lo).astype(int)
    crop=cv2.warpAffine(crop,mat,(rw,rh));mask=cv2.warpAffine(mask,mat,(rw,rh));yy,xx=np.where(mask>127)
    if not len(xx) or max(rw,rh)>=size-4:continue
    x=int(rng.integers(2,size-rw-1));y=int(rng.integers(2,size-rh-1));b=[x+xx.min(),y+yy.min(),x+xx.max()+1,y+yy.max()+1]
    if any(min(b[2],p[3])>max(b[0],p[1]) and min(b[3],p[4])>max(b[1],p[2]) for p in boxes):continue
    # Photometric variation is independent for object and ground.
    if self.domain_randomization:
     color=crop.astype(np.float32)/255;gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)[:,:,None]
     color=gray+(color-gray)*rng.uniform(.3,1.5)
     color=np.clip(color,0,1)**rng.uniform(.65,1.6)
     crop=np.clip((color*rng.uniform(.45,1.5)+rng.uniform(-.06,.06))*255,0,255).astype(np.uint8)
    else:crop=np.clip(crop.astype(float)*rng.uniform(.65,1.2)+rng.uniform(-20,15),0,255).astype(np.uint8)
    alpha=mask.astype(np.float32)[:,:,None]/255;image[y:y+rh,x:x+rw]=(crop*alpha+image[y:y+rh,x:x+rw]*(1-alpha)).astype(np.uint8)
    pad=5 if self.classes[c]=='small_launcher' else max(2,round(.08*max(b[2]-b[0],b[3]-b[1])))
    boxes.append([c,max(0,b[0]-pad),max(0,b[1]-pad),min(size,b[2]+pad),min(size,b[3]+pad)])
  if rng.random()<.5:
   image=image[:,::-1].copy();boxes=[[c,size-x2,y1,size-x1,y2]for c,x1,y1,x2,y2 in boxes];partial=[[c,size-x2,y1,size-x1,y2]for c,x1,y1,x2,y2 in partial]
  if rng.random()<.5:
   image=image[::-1].copy();boxes=[[c,x1,size-y2,x2,size-y1]for c,x1,y1,x2,y2 in boxes];partial=[[c,x1,size-y2,x2,size-y1]for c,x1,y1,x2,y2 in partial]
  image=np.clip(image.astype(float)*rng.uniform(.8,1.15)+rng.uniform(-10,10),0,255).astype(np.uint8)
  dim=size//4;heat=np.zeros((len(self.classes),dim,dim),np.float32);target=np.zeros((4,dim,dim),np.float32);valid=np.zeros((1,dim,dim),np.float32)
  for c,x1,y1,x2,y2 in partial:heat[:,max(0,int(y1/4)):min(dim,int(np.ceil(y2/4))+1),max(0,int(x1/4)):min(dim,int(np.ceil(x2/4))+1)]=-1
  for c,x1,y1,x2,y2 in boxes:
   cx=(x1+x2)/8;cy=(y1+y2)/8;ix=min(dim-1,int(cx));iy=min(dim-1,int(cy));radius=max(1,int(min(x2-x1,y2-y1)/16));sigma=max(.8,radius/2)
   l=max(0,ix-radius);r=min(dim,ix+radius+1);t=max(0,iy-radius);b=min(dim,iy+radius+1);ys,xs=np.mgrid[t:b,l:r];g=np.exp(-((xs-ix)**2+(ys-iy)**2)/(2*sigma*sigma));heat[c,t:b,l:r]=np.maximum(heat[c,t:b,l:r],g);heat[c,iy,ix]=1
   target[:,iy,ix]=[np.log(max(1,x2-x1)/32),np.log(max(1,y2-y1)/32),cx-ix,cy-iy];valid[:,iy,ix]=1
  return image.transpose(2,0,1).copy().astype(np.float32)/255,heat,target,valid
