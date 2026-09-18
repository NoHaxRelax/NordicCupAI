"""Separate background review batch, preserving the first dataset and its split."""
import json
from pathlib import Path
import cv2
from .build import ROOT,digest,write,render,intersection

def build():
 base=ROOT/'data/drone/grid384-20260918-v1';out=base/'batch2';out.mkdir(exist_ok=False)
 previous=json.loads((base/'manifest-approved.json').read_text())
 existing={(r['frame'],tuple(r['source_rect_xyxy'])) for r in previous['records'] if r['source']=='validation'}
 # Regions selected visually from native flight overviews; no emptiness inferred.
 zones=[(65,'train','buildings',(1920,384,3456,1536),12),(85,'train','buildings',(1920,960,3456,1776),12),(125,'dev','dense_buildings',(0,0,1920,768),8),(145,'dev','dense_buildings',(1920,0,3456,1152),8),(105,'dev','water_shoreline',(0,0,3072,192),4),(125,'dev','water_shoreline',(384,1152,3072,1536),4)]
 complete={r['frame'] for r in json.loads((ROOT/'data/drone/reconstructed-validation/manifest.json').read_text()) if r.get('complete') and r.get('native_coverage')==1}
 records=[];candidates=[];sources={}
 for f,split,scene,(xmin,ymin,xmax,ymax),count in zones:
  assert f in complete
  ip=ROOT/f'data/drone/reconstructed-validation/frame_{f:06d}.png';ap=ROOT/f'data/drone/training/score-anchored-validation-v8/finetune-metadata/annotations/frame_{f:06d}.json'
  im=cv2.imread(str(ip));aa=json.loads(ap.read_text())['annotations'];sources[str(ip.relative_to(ROOT))]=digest(ip);sources[str(ap.relative_to(ROOT))]=digest(ap)
  choices=[]
  for y in range(ymin,ymax+1,384):
   for x in range(xmin,xmax+1,384):
    box=[x,y,x+384,y+384]
    if box[2]>3840 or box[3]>2160 or (f,tuple(box)) in existing:continue
    if any(intersection([x-64,y-64,x+448,y+448],a['bbox']) for a in aa):continue
    choices.append(box)
  # Spread across region, rather than selecting a contiguous corner.
  import numpy as np
  selected=[choices[i] for i in np.linspace(0,len(choices)-1,min(count,len(choices))).round().astype(int)]
  for box in selected:
   x,y,u,v=box;tid=f'batch2-validation-f{f:06d}-x{x:04d}-y{y:04d}'
   for z in range(3):
    file=f'images/{tid}-L{z}.png';p=out/file;p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),render(im[y:v,x:u],z))
    records.append(dict(id=tid+f'-L{z}',tile_id=tid,source='validation',frame=f,source_file=str(ip.relative_to(ROOT)),source_sha256=digest(ip),source_rect_xyxy=box,zoom=z,native_crop_size=[96,192,384][z],input_size=384,file=file,sha256=digest(p),split='pending_review',intended_split=split,scene_type=scene,kind='unknown',annotations=[],target_class_ids=[],annotation_complete=False,supervision='none',background_target=None,eligible_for_training=False))
   candidates.append(dict(tile_id=tid,frame=f,source_rect_xyxy=box,source_sha256=records[-1]['source_sha256'],native_image=records[-1]['file'],native_image_sha256=records[-1]['sha256'],status='unreviewed',intended_split=split,scene_type=scene))
 m=dict(schema=1,classes=previous['classes'],parent_manifest_sha256=digest(base/'manifest-approved.json'),source_hashes=sources,records=records,policy='Addendum only. Pending human review. Train candidates <=90; dev candidates >=100. Do not train on dev negatives. No known-box overlap including projected labels and 64px margin. No guarantee of emptiness.')
 write(out/'manifest.json',m);d=dict(dataset_manifest_sha256=digest(out/'manifest.json'),candidates=candidates);write(out/'review-candidates.json',d);(out/'review-data.js').write_text('window.REVIEW_DATA='+json.dumps(d)+';')
 page=(ROOT/'drone/grid_dataset/review.html').read_text().replace('Check candidate empty squares','Batch 2: buildings and water').replace('Review validation background squares','Batch 2 background review').replace('nothing enters training','nothing enters training')
 page=page.replace("source box ['+r.source_rect_xyxy.join(', ')+']'","source box ['+r.source_rect_xyxy.join(', ')+'] · '+r.scene_type+' · '+r.intended_split.toUpperCase()")
 page=page.replace('drone-grid-empty-review.json','drone-grid-empty-review-batch2.json').replace('These 384×384 native-pixel validation squares','TRAIN squares supplement training; DEV squares are reserved for measuring false positives. These 384×384 native-pixel validation squares')
 (out/'review.html').write_text(page)
 from collections import Counter
 summary=dict(squares=len(candidates),images=len(records),counts=dict(Counter(r['intended_split']+'/'+r['scene_type'] for r in candidates)))
 write(out/'summary.json',summary);print(json.dumps(summary))
 # Contact sheet for content QA, not human emptiness approval.
 from PIL import Image,ImageDraw
 sheet=Image.new('RGB',(8*160,((len(candidates)+7)//8)*185));draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(candidates):
  x=i%8*160;y=i//8*185;img=Image.open(out/r['native_image']);img.thumbnail((160,160));sheet.paste(img,(x,y+25));draw.text((x,y),f'{i+1} f{r["frame"]} {r["intended_split"]}',fill='white')
 sheet.save(out/'contact.jpg')
if __name__=='__main__':build()
