"""Refresh existing validation crops from the current corrected label snapshot."""
import json,shutil,math
from pathlib import Path
import cv2,numpy as np
from .detector import sha


def build(root,source,output):
 shutil.copytree(source,output);m=json.loads((source/'manifest.json').read_text());kept=[];audit=[];previews=[]
 for old in m['templates']:
  if old['source']=='organizer_reference':kept.append(old);continue
  p=root/f"data/drone/training/score-anchored-validation-v7/{old['group']}.json";doc=json.loads(p.read_text());a=next((a for a in doc.get('annotations',[])if a['frame']==old['frame']),None)
  if a is None:audit.append(dict(id=old['id'],status='removed: no longer a retained annotation'));continue
  assert a['class']==old['class']
  b=[math.floor(v) for v in a['bbox_source_xyxy'][:2]]+[math.ceil(v)for v in a['bbox_source_xyxy'][2:]]
  if not 0<b[0]<b[2]<3839 or not 0<b[1]<b[3]<2159:audit.append(dict(id=old['id'],status='removed: incomplete crop'));continue
  path=root/f"data/drone/reconstructed-validation/frame_{old['frame']:06d}.png";image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);crop=image[b[1]:b[3],b[0]:b[2]];assert crop.shape[2]==3 or np.all(crop[:,:,3]==255);crop=crop[:,:,:3];h,w=crop.shape[:2]
  if old.get('mask_provenance')=='manual calibration silhouette' and b==old['bbox']:mask=cv2.imread(str(source/old['mask_file']),0)
  else:
   gc=np.zeros((h,w),np.uint8);cv2.setRNGSeed(0);cv2.grabCut(crop,gc,(1,1,w-2,h-2),np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_RECT);mask=np.isin(gc,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255;mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
  row={k:v for k,v in old.items()if k!='mask_fallback'};row.update(bbox=b,annotation_extent=True,annotation_file=str(p.relative_to(root)),annotation_sha256=sha(p),review_status=a['review_status'],foreground_fraction=float(np.mean(mask>0)))
  if np.count_nonzero(mask)<8:row['mask_fallback']='No stable foreground from corrected crop';mask[:]=255
  dest=output/row['file'];mp=output/row['mask_file'];cv2.imwrite(str(dest),crop);cv2.imwrite(str(mp),mask);row.update(sha256=sha(dest),mask_sha256=sha(mp));kept.append(row)
  m['source_hashes'][str(p.relative_to(root))]=sha(p);m['source_hashes'][str(path.relative_to(root))]=sha(path);audit.append(dict(id=row['id'],status='updated',old_bbox=old['bbox'],new_bbox=b,mask_fallback=row.get('mask_fallback')))
  canvas=np.full((190,360,3),235,np.uint8)
  for offset,pic in [(0,crop),(180,cv2.bitwise_and(crop,crop,mask=mask))]:
   f=min(175/w,155/h);pic=cv2.resize(pic,None,fx=f,fy=f);canvas[30:30+pic.shape[0],offset:offset+pic.shape[1]]=pic
  cv2.putText(canvas,row['id'],(2,18),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1);previews.append(canvas)
 m['templates']=kept;m['classes']={c:sum(r['class']==c for r in kept)for c in m['classes']};m['validation_templates']=sum(r['source']!='organizer_reference'for r in kept);m['label_refresh']=dict(annotation_snapshot='score-anchored-validation-v7',base_manifest_sha256=sha(source/'manifest.json'),new_validation_frames=0,audit=audit)
 (output/'manifest.json').write_text(json.dumps(m,indent=2));cv2.imwrite(str(output/'corrected-crop-audit.jpg'),np.vstack([np.hstack(previews[i:i+3]+[np.full((190,360,3),235,np.uint8)]*(3-len(previews[i:i+3])))for i in range(0,len(previews),3)]))
 print(json.dumps(dict(templates=len(kept),validation_templates=m['validation_templates'],removed=[r for r in audit if r['status']!='updated'])))


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.root,a.source,a.output)
