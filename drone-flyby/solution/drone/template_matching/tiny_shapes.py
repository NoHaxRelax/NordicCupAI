"""Fixed-asset color and silhouette matching for the tiny green launcher."""
import json
from pathlib import Path
import cv2
import numpy as np
from .detector import TemplateDetector,sha,nms


class TinyShapeDetector(TemplateDetector):
    def __init__(self,bank,threshold=.65,min_contrast=.2):
        if not 0<=threshold<=1 or not 0<=min_contrast<=1:raise ValueError('Invalid similarity threshold')
        self.threshold=threshold;self.min_contrast=min_contrast;self.bank=Path(bank)
        self.manifest=json.loads((self.bank/'manifest.json').read_text())
        self.patterns=[];pixels=[];contrasts=[]
        for row in self.manifest['templates']:
            if row['class']!='small_launcher':continue
            path=self.bank/row['file'];assert sha(path)==row['sha256']
            image=cv2.imread(str(path));lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
            # Asset's green paint is distinct from the pink reference ground.
            mask=((lab[:,:,1]<126)&(lab[:,:,2]>140)).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
            if count<2:continue
            label=1+np.argmax(stats[1:,4]);mask=(labels==label).astype(np.uint8)
            pixels.extend(lab[mask>0]);x,y,w,h,_=stats[label]
            border=np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])
            contrasts.append(np.median(border,axis=0)[1:]-np.median(lab[mask>0],axis=0)[1:])
            patch=mask[y:y+h,x:x+w]*255
            for angle in range(0,360,15):
                canvas=np.zeros((64,64),np.uint8);f=36/max(w,h);resized=cv2.resize(patch,None,fx=f,fy=f,interpolation=cv2.INTER_NEAREST);rh,rw=resized.shape;canvas[(64-rh)//2:(64-rh)//2+rh,(64-rw)//2:(64-rw)//2+rw]=resized
                rot=cv2.warpAffine(canvas,cv2.getRotationMatrix2D((31.5,31.5),angle,1),(64,64),flags=cv2.INTER_NEAREST)
                yy,xx=np.where(rot>0);cropped=rot[yy.min():yy.max()+1,xx.min():xx.max()+1]
                self.patterns.append((row['id'],angle,cv2.resize(cropped,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)/255,cropped.shape[1]/cropped.shape[0]))
        if not self.patterns:raise ValueError('Bank has no usable small-launcher shapes')
        self.contrast=np.median(contrasts,axis=0)
        p=np.asarray(pixels);self.center=np.median(p,axis=0);self.spread=np.maximum(np.std(p,axis=0),[16,4,5])
        self.masks=np.array([r[2] for r in self.patterns]);self.aspect=np.array([r[3] for r in self.patterns])

    def detect(self,image,pixels_per_source_pixel=1.):
        if image is None or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected uint8 BGR image')
        if not np.isfinite(pixels_per_source_pixel) or pixels_per_source_pixel<=0:
            raise ValueError('Invalid image scale')
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
        z=(lab-self.center)/self.spread
        probability=np.exp(-.5*(z[:,:,1]**2+z[:,:,2]**2))
        mask=(probability>.35).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        rows=[];s=pixels_per_source_pixel
        for label in range(1,count):
            x,y,w,h,area=stats[label]
            if not 6*s<=w<=24*s or not 6*s<=h<=28*s or not 25*s*s<=area<=350*s*s:continue
            if x==0 or y==0 or x+w==image.shape[1] or y+h==image.shape[0]:continue
            crop=(labels[y:y+h,x:x+w]==label).astype(np.float32)
            target=cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA)
            intersect=np.minimum(self.masks,target).sum((1,2));union=np.maximum(self.masks,target).sum((1,2))
            shapes=intersect/np.maximum(union,1)*np.exp(-.8*np.abs(np.log(self.aspect/(w/h))))
            best=int(shapes.argmax());color=float(probability[y:y+h,x:x+w][crop>0].mean())
            # Shape/color are asset likelihood indicators, not calibrated probabilities.
            radius=max(3,round(5*s));left=max(0,x-radius);top=max(0,y-radius);right=min(image.shape[1],x+w+radius);bottom=min(image.shape[0],y+h+radius)
            surround=lab[top:bottom,left:right];ring=np.ones(surround.shape[:2],bool);ring[y-top:y+h-top,x-left:x+w-left]=False
            delta=np.median(surround[ring],axis=0)[1:]-np.median(lab[y:y+h,x:x+w][crop>0],axis=0)[1:]
            contrast_score=float(np.exp(-.5*np.sum(((delta-self.contrast)/5)**2)))
            score=float(.65*shapes[best]+.15*color+.2*contrast_score)
            if contrast_score<self.min_contrast:continue
            if score<self.threshold:continue
            template,angle,_,_=self.patterns[best]
            pad=max(1,round(5*s))
            rows.append(dict(**{'class':'small_launcher'},bbox=[max(0,int(x)-pad),max(0,int(y)-pad),min(image.shape[1],int(x+w)+pad),min(image.shape[0],int(y+h)+pad)],score=score,template_id=template,angle=angle,shape_similarity=float(shapes[best]),color_similarity=color,contrast_similarity=contrast_score,method='tiny_shape'))
        return nms(rows,.35,300)
