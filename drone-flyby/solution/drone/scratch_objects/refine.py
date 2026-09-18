"""Optional deterministic foreground refinement of CNN small-launcher boxes.

Palette prototypes come only from bank foreground/background masks. This does
not change class or confidence and is evaluated separately from raw CNN output.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from drone.template_matching.detector import sha


class PixelRefiner:
    def __init__(self,bank,classes=('small_launcher',)):
        bank=Path(bank);manifest=json.loads((bank/'manifest.json').read_text());self.palettes={}
        for label in classes:
            fg=[];bg=[]
            for row in manifest['templates']:
                if row['class']!=label or row.get('mask_fallback'):continue
                for k,h in [('file','sha256'),('mask_file','mask_sha256')]:
                    if sha(bank/row[k])!=row[h]:raise ValueError('Refinement bank hash mismatch')
                image=cv2.imread(str(bank/row['file']));mask=cv2.imread(str(bank/row['mask_file']),0)
                core=cv2.erode(mask,np.ones((5,5),np.uint8))>0
                background=cv2.dilate(mask,np.ones((3,3),np.uint8))==0
                colors=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
                if core.sum()>=8:fg.extend(colors[core])
                if background.sum()>=8:bg.extend(colors[background])
            if len(fg)<16 or len(bg)<16:continue
            def prototypes(values):
                values=np.array(values,np.float32)
                if len(values)>10000:values=values[np.linspace(0,len(values)-1,10000).astype(int)]
                cv2.setRNGSeed(1731)
                return cv2.kmeans(values,8,None,(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,50,.1),3,cv2.KMEANS_PP_CENTERS)[2]
            self.palettes[label]=(prototypes(fg),prototypes(bg))

    def refine(self,image,rows):
        output=[]
        for row in rows:
            if row['class'] not in self.palettes:output.append(row);continue
            x1,y1,x2,y2=row['bbox'];x1=max(0,int(np.floor(x1)));y1=max(0,int(np.floor(y1)))
            x2=min(image.shape[1],int(np.ceil(x2)));y2=min(image.shape[0],int(np.ceil(y2)))
            crop=image[y1:y2,x1:x2,:3];h,w=crop.shape[:2]
            if min(h,w)<8 or max(h,w)>160:output.append(row);continue
            lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(np.float32)
            fg,bg=self.palettes[row['class']]
            fd=((lab[:,:,None,:]-fg)**2).sum(3).min(2)
            bd=((lab[:,:,None,:]-bg)**2).sum(3).min(2)
            mask=((fd<.75*bd)&(fd<625)).astype(np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            count,components,stats,centers=cv2.connectedComponentsWithStats(mask,8)
            candidates=[]
            for index in range(1,count):
                x,y,bw,bh,area=stats[index]
                distance=np.linalg.norm((centers[index]-[w/2,h/2])/[w,h])
                if area>=8 and area/(h*w)>=.05 and distance<.35:
                    candidates.append((area/(1+distance*4),x,y,bw,bh,area))
            if not candidates:output.append(row);continue
            _,x,y,bw,bh,area=max(candidates)
            padding=2
            box=[float(x1+max(0,x-padding)),float(y1+max(0,y-padding)),float(x1+min(w,x+bw+padding)),float(y1+min(h,y+bh+padding))]
            output.append({**row,'bbox':box,'unrefined_bbox':row['bbox'],'refinement':'bank_Lab_foreground','foreground_pixels':int(area)})
        return output
