"""GPU masked normalized correlation against explicitly calibrated asset poses.

Search uses image pixels alone. FFT correlations give exactly the same valid
translation search as spatial masked correlation; no target coordinates enter.
"""
import cv2
import numpy as np
import torch
from .detector import TemplateDetector,Settings,features,correlation,nms


def masked_ncc_fft(scene,templates,masks):
    """Return batched valid NCC maps, for equal-size templates (N,C,h,w)."""
    shape=scene.shape[-2:];h,w=templates.shape[-2:]
    sf=torch.fft.rfft2(scene,s=shape);sqf=torch.fft.rfft2(scene.square(),s=shape)
    mf=torch.fft.rfft2(masks,s=shape);count=masks.sum((-2,-1),keepdim=True)
    centered=(templates-(templates*masks).sum((-2,-1),keepdim=True)/count)*masks
    tf=torch.fft.rfft2(centered,s=shape)
    numerator=torch.fft.irfft2(sf*tf.conj(),s=shape)
    total=torch.fft.irfft2(sf*mf.conj(),s=shape)
    total2=torch.fft.irfft2(sqf*mf.conj(),s=shape)
    energy=(total2-total.square()/count).clamp_min(0)
    denominator=(energy*centered.square().sum((-2,-1),keepdim=True)).sqrt()
    result=torch.where(denominator>1e-6,numerator/denominator.clamp_min(1e-6),0)
    return result[...,:shape[0]-h+1,:shape[1]-w+1].clamp(-1,1)


class CalibratedPixelDetector(TemplateDetector):
    def __init__(self,bank,threshold=.72,device='cpu'):
        super().__init__(bank,Settings(scales=(.85,1.,1.15),angles=(-8.,0.,8.),proposal_threshold=.55,score_threshold=threshold,peaks_per_template=2,mask_mode='masked_ncc'))
        self.templates=[t for t in self.templates if t[0].get('calibration')]
        if not self.templates:raise ValueError('No calibrated asset poses')
        self.device=torch.device(device)

    def detect(self,image,pixels_per_source_pixel=1.):
        if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected uint8 BGR image')
        gray,high=features(image);shape=gray.shape
        scene=torch.as_tensor(np.stack([gray-.5,high]),device=self.device)
        sf=torch.fft.rfft2(scene);sqf=torch.fft.rfft2(scene.square())
        variants=self.variants(pixels_per_source_pixel);rows=[]
        # Process small groups to keep memory bounded even on large input crops.
        with torch.inference_mode():
            for first in range(0,len(variants),6):
                group=[v for v in variants[first:first+6] if v[4].shape[0]<=shape[0] and v[4].shape[1]<=shape[1]]
                if not group:continue
                kernels=np.zeros((len(group),2,*shape),np.float32);masks=np.zeros((len(group),1,*shape),np.float32);counts=[];energies=[]
                for i,(_,_,_,_,tg,th,mask) in enumerate(group):
                    h,w=tg.shape;values=np.stack([tg,th]);centered=(values-values[:,mask].mean(1)[:,None,None])*mask
                    kernels[i,:,:h,:w]=centered;masks[i,0,:h,:w]=mask;counts.append(mask.sum());energies.append((centered**2).sum((1,2)))
                tf=torch.fft.rfft2(torch.as_tensor(kernels,device=self.device));mf=torch.fft.rfft2(torch.as_tensor(masks,device=self.device))
                count=torch.as_tensor(counts,device=self.device)[:,None,None,None];energy=torch.as_tensor(np.array(energies),device=self.device)[:,:,None,None]
                numerator=torch.fft.irfft2(sf*tf.conj(),s=shape);total=torch.fft.irfft2(sf*mf.conj(),s=shape);total2=torch.fft.irfft2(sqf*mf.conj(),s=shape)
                denom=((total2-total.square()/count).clamp_min(0)*energy).sqrt()
                maps=torch.where(denom>1e-6,numerator/denom.clamp_min(1e-6),0).clamp(-1,1)
                responses=(.45*maps[:,0]+.55*maps[:,1]).cpu().numpy()
                for variant,response in zip(group,responses):
                    row,scale,angle,patch,tg,th,mask=variant;h,w=tg.shape;response=response[:shape[0]-h+1,:shape[1]-w+1].copy()
                    for _ in range(self.settings.peaks_per_template):
                        _,peak,_,(x,y)=cv2.minMaxLoc(response)
                        if peak<self.settings.proposal_threshold:break
                        color=correlation(image[y:y+h,x:x+w][mask].reshape(-1,1,3),patch[mask].reshape(-1,1,3));score=float(np.clip(.75*peak+.25*max(0,color),0,1))
                        if score>=self.settings.score_threshold:rows.append(dict(**{'class':row['class']},bbox=[x,y,x+w,y+h],score=score,pixel_correlation=color,template_id=row['id'],scale=scale,angle=angle,family='calibrated_pixels'))
                        rx,ry=max(1,w//3),max(1,h//3);response[max(0,y-ry):y+ry+1,max(0,x-rx):x+rx+1]=-1
        return nms(rows,self.settings.nms_iou,self.settings.max_detections)
