"""Audit camera-boundary clipping against frozen source annotations."""
from pathlib import Path
from collections import Counter,defaultdict
from common import read,sha,write
from evaluate_views import iou
root=Path('.');data=Path('data/drone/overnight-runpod-20260918-v2');m=read(data/'manifest.json');aliases=m['config']['track_aliases'];lookup={}
for file,expected in m['source_hashes'].items():
 p=Path(file)
 if not p.name.endswith('.json') or 'algorithmic-full-validation' not in file:continue
 assert sha(p)==expected;doc=read(p)
 if 'track_id' not in doc:continue
 track=aliases.get(doc['track_id'],doc['track_id'])
 for a in doc.get('annotations',[]):
  if not a.get('review_status','').startswith('direct'):continue
  key=('validation',a['frame'],track);value=(a['class'],a['bbox_source_xyxy'])
  if key in lookup:assert lookup[key]==value
  lookup[key]=value
for file,expected in m['source_hashes'].items():
 if '/reference/' not in file or '/annotations/' not in file:continue
 p=Path(file);assert sha(p)==expected;doc=read(p)
 for a in doc['annotations']:lookup['reference',doc['frame'],'reference:'+a['object_id']]=(a['object_id'],a['bbox'])
summary=defaultdict(Counter);per_class=defaultdict(Counter);records=[]
for row in m['records']:
 if row['task']!='detector':continue
 w,h=row['dimensions'];region=row['source_region'];scale=(region[2]-region[0])/w;lines=(data/row['label_file']).read_text().splitlines();assert len(lines)==len(row['tracks']);bad=[]
 for line,track in zip(lines,row['tracks']):
  cls,x,y,bw,bh=map(float,line.split());label=m['classes'][int(cls)];expected_label,box=lookup[row['source'],row['frame'],track];assert label==expected_label
  visible=[max(box[0],region[0]),max(box[1],region[1]),min(box[2],region[2]),min(box[3],region[3])]
  actual=[region[0]+(x-bw/2)*w*scale,region[1]+(y-bh/2)*h*scale,region[0]+(x+bw/2)*w*scale,region[1]+(y+bh/2)*h*scale];assert iou(visible,actual)>.999
  fraction=(visible[2]-visible[0])*(visible[3]-visible[1])/((box[2]-box[0])*(box[3]-box[1]));clipped=fraction<.99999;summary[row['split']]['boxes']+=1;summary[row['split']]['clipped_boxes']+=int(clipped);summary[row['split']]['under_half_visible']+=int(fraction<.5)
  if row['split']=='dev':per_class[label]['boxes']+=1;per_class[label]['clipped']+=int(clipped);per_class[label]['under_half']+=int(fraction<.5)
  if clipped:bad.append(dict(track=track,class_name=label,visible_fraction=fraction,full_bbox=box,visible_bbox=visible))
 summary[row['split']]['views']+=1;summary[row['split']]['views_with_clipped_target']+=bool(bad)
 records.append(dict(file=row['file'],split=row['split'],clipped=bad))
out=Path('artifacts/drone-overnight-runpod/clipped-view-audit.json');write(out,dict(summary=summary,dev_per_class=per_class,records=records,source_fingerprint=m['source_fingerprint'],limitations=['Clipping is a camera-view construction property, not evidence of incorrect source annotations. Original experiment results are preserved.']))
print(dict(summary));print(dict(per_class))
