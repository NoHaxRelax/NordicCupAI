from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from drone.template_matching.detector import TemplateDetector,nms,sha
from .model import AssetNet


class AssetHeatmapDetector(TemplateDetector):
 def __init__(self,weights,threshold=.2,device='cpu'):
  self.device=device;self.threshold=threshold;self.checkpoint_sha256=sha(Path(weights));checkpoint=torch.load(weights,map_location='cpu',weights_only=True);self.classes=checkpoint['classes'];self.model=AssetNet(len(self.classes));self.model.load_state_dict(checkpoint['state_dict']);self.model.to(device).eval()
 def detect(self,image,pixels_per_source_pixel=1.):
  if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Expected BGR image')
  h,w=image.shape[:2];padded=cv2.copyMakeBorder(image,0,(-h)%16,0,(-w)%16,cv2.BORDER_REFLECT)
  with torch.inference_mode():
   heat,box=self.model(torch.from_numpy(padded.transpose(2,0,1).copy()).float().unsqueeze(0).to(self.device)/255);prob=heat.sigmoid();peak=(F.max_pool2d(prob,3,1,1)==prob)&(prob>=self.threshold);indices=peak[0].nonzero();scores=prob[0][peak[0]];order=scores.argsort(descending=True)[:100];indices=indices[order].cpu().numpy();scores=scores[order].cpu().numpy();box=box[0].cpu().numpy()
  rows=[]
  for (c,y,x),score in zip(indices,scores):
   bw,bh=np.exp(np.clip(box[:2,y,x],-3,4))*32;dx,dy=np.clip(box[2:,y,x],0,1);cx=(x+dx)*4;cy=(y+dy)*4
   if cx>=w or cy>=h:continue
   rows.append(dict(**{'class':self.classes[c]},score=float(score),bbox=[float(max(0,cx-bw/2)),float(max(0,cy-bh/2)),float(min(w,cx+bw/2)),float(min(h,cy+bh/2))],template_id='asset_heatmap',family='heatmap'))
  return nms(rows,.35,100)
