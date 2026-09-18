"""Add six pose crops from two disclosed calibration frames."""
import json,shutil
from pathlib import Path
import cv2,numpy as np
from .detector import sha


def build(root,source,output):
 shutil.copytree(source,output);m=json.loads((source/'manifest.json').read_text());selected=[('jet-plane-e-109-141',110,'jet_plane'),('helicopter-b-105-149',110,'helicopter'),('large-launcher-b-139-171',150,'large_launcher'),('tank-b-128-160',150,'tank'),('medium-launcher-a-092-123',110,'medium_launcher'),('large-tower-b-092-123',110,'large_tower')];images=[]
 for track,frame,label in selected:
  ann=root/f'data/drone/training/score-anchored-validation-v7/{track}.json';doc=json.loads(ann.read_text());a=next(a for a in doc['annotations']if a['frame']==frame)
  assert a['class']==label and (a['review_status'].startswith('direct') or (label=='large_launcher' and a['review_status']=='score_anchored_track_projection'))
  b=list(map(int,a['bbox_source_xyxy']));assert 0<b[0]<b[2]<3839 and 0<b[1]<b[3]<2159
  path=root/f'data/drone/reconstructed-validation/frame_{frame:06d}.png';image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);crop=image[b[1]:b[3],b[0]:b[2]];assert crop.shape[2]==3 or np.all(crop[:,:,3]==255);crop=crop[:,:,:3];h,w=crop.shape[:2]
  gc=np.zeros((h,w),np.uint8);cv2.setRNGSeed(0);cv2.grabCut(crop,gc,(1,1,w-2,h-2),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT);mask=np.isin(gc,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
  if label=='jet_plane':
   # Audited on the calibration crop only. The gray wings touch similarly
   # colored concrete, which makes rectangle-initialized GrabCut collapse.
   polygon=np.array([[30,2],[31,11],[34,12],[38,11],[35,16],[32,19],[33,26],[53,23],[53,28],[48,32],[34,38],[31,56],[29,57],[27,49],[26,39],[6,30],[4,22],[25,27],[26,18],[23,16],[18,10],[22,12],[26,14],[29,10]],np.int32)
   mask[:]=0;cv2.fillPoly(mask,[polygon],255)
  assert np.count_nonzero(mask)>=8,(label,np.count_nonzero(mask))
  mask=cv2.dilate(mask,np.ones((3,3),np.uint8));name=f'pose-{label}-{frame:06d}';dest=output/(name+'.png');mp=output/(name+'-mask.png');cv2.imwrite(str(dest),crop);cv2.imwrite(str(mp),mask)
  m['templates'].append(dict(id=name,file=dest.name,sha256=sha(dest),mask_file=mp.name,mask_sha256=sha(mp),foreground_fraction=float(np.mean(mask>0)),**{'class':label},frame=frame,bbox=b,source='reviewed_validation',group=track,calibration=True,annotation_extent=True,mask_provenance='manual calibration silhouette' if label=='jet_plane' else 'GrabCut calibration crop'))
  m['source_hashes'][str(path.relative_to(root))]=sha(path);m['source_hashes'][str(ann.relative_to(root))]=sha(ann);m['classes'][label]+=1
  canvas=np.full((240,480,3),240,np.uint8)
  for offset,pic in [(0,crop),(240,cv2.bitwise_and(crop,crop,mask=mask))]:
   f=min(220/w,190/h);p=cv2.resize(pic,None,fx=f,fy=f);canvas[40:40+p.shape[0],offset:offset+p.shape[1]]=p
  cv2.putText(canvas,name,(3,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);images.append(canvas)
 m['pose_calibration']=dict(frames=[110,150],templates=6,selection=selected,base_manifest_sha256=sha(source/'manifest.json'),policy='Additional allowed validation use. Same physical tracks as development: results measure calibrated fixed-asset recognition, not unseen-track generalization. Original whole-track cutoff applies only to base bank. No late 181-249 images are used for training or calibration.')
 m['validation_templates']+=6;(output/'manifest.json').write_text(json.dumps(m,indent=2));cv2.imwrite(str(output/'pose-audit.jpg'),np.vstack(images))


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.root,a.source,a.output)
