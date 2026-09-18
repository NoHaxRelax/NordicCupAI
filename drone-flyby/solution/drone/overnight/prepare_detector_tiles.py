"""Add overlapping native-zoom TRAIN tiles; keep every dev record unchanged."""
import argparse,copy,os,tarfile,time
from collections import Counter
from pathlib import Path
from common import read,write,sha,fingerprint,verify_dataset

def starts(length,size):return sorted(set(list(range(0,max(1,length-size+1),max(1,size//2)))+[max(0,length-size)]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);a.artifacts.mkdir(exist_ok=True,parents=True)
 from PIL import Image
 m=read(a.source/'manifest.json');verify_dataset(a.source,m);original=copy.deepcopy(m)
 for r in m['records']:
  for key in ['file','label_file']:
   if key not in r:continue
   dest=a.output/r[key]
   if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);os.link(a.source/r[key],dest)
 extra=[];files=[];skipped=0;started=time.monotonic();views=[r for r in m['records'] if r['task']=='detector' and r['split']=='train' and r['zoom']==2]
 for i,r in enumerate(views):
  if r['source']=='validation':assert r['frame']<=164
  im=Image.open(a.source/r['file']).convert('RGB');w,h=im.size;tw,th=min(w,480),min(h,270);boxes=[]
  lines=(a.source/r['label_file']).read_text().splitlines();assert len(lines)==len(r['tracks'])
  for j,l in enumerate(lines):
   c,x,y,bw,bh=map(float,l.split());boxes.append((int(c),max(0,(x-bw/2)*w),max(0,(y-bh/2)*h),min(w,(x+bw/2)*w),min(h,(y+bh/2)*h),r['tracks'][j]))
  for x in starts(w,tw):
   for y in starts(h,th):
    labels=[];classes=[];tracks=[];partial=False
    for c,x1,y1,x2,y2,track in boxes:
     if min(x+tw,x2)<=max(x,x1) or min(y+th,y2)<=max(y,y1):continue
     if x1<x or y1<y or x2>x+tw or y2>y+th:partial=True;break
     labels.append(f'{c} {((x1+x2)/2-x)/tw:.8f} {((y1+y2)/2-y)/th:.8f} {(x2-x1)/tw:.8f} {(y2-y1)/th:.8f}')
     classes.append(m['classes'][c]);tracks.append(track)
    if partial:skipped+=1;continue
    name=f'native-tile-{i:05d}-{x}-{y}';image=f'detector/images/train/{name}.png';label=f'detector/labels/train/{name}.txt';im.crop((x,y,x+tw,y+th)).save(a.output/image);(a.output/label).write_text('\n'.join(labels)+('\n' if labels else ''))
    region=r['source_region'];extra.append(dict(task='detector',split='train',source=r['source'],frame=r['frame'],zoom=2,source_region=[region[0]+x,region[1]+y,region[0]+x+tw,region[1]+y+th],file=image,sha256=sha(a.output/image),dimensions=[tw,th],label_file=label,label_sha256=sha(a.output/label),classes=classes,tracks=tracks,parent_file=r['file'],parent_sha256=r['sha256'],tile_bounds=[x,y,x+tw,y+th]));files.extend([image,label])
  if i%25==0:write(a.artifacts/'preparation-progress.json',dict(source_views=i+1,total=len(views),added=len(extra),skipped_partial=skipped))
 m['records']+=extra;assert [r for r in m['records'] if r['split']=='dev']==[r for r in original['records'] if r['split']=='dev']
 m['config']['experiments']=[dict(id='det-tiles-medium',detector_model='yolo26m',task='detector',initialization='pretrained',adaptation='full',zoom='mixed',epochs=80,lr=.0003),dict(id='det-tiles-large',detector_model='yolo26x',task='detector',initialization='pretrained',adaptation='full',zoom='mixed',epochs=60,lr=.0003)]
 m['derivation']=dict(source_manifest_sha256=sha(a.source/'manifest.json'),native_tile=[480,270],overlap=.5,partial_targets='skip whole training tile',code_sha256=sha(Path(__file__)))
 m['source_fingerprint']=fingerprint(dict(derivation=m['derivation'],config=m['config']));m['counts']=dict(Counter(r['task']+'/'+r['split'] for r in m['records']));m['limitations']+=['Overlapping tiles augment TRAIN only. Checkpoint selection uses original full dev views. Blind tiled inference must be evaluated separately on original view labels.'];write(a.output/'manifest.json',m);write(a.output/'config.json',m['config']);verify_dataset(a.output,m)
 release=read(a.release);assert release['source_fingerprint']==original['source_fingerprint']
 release['source_fingerprint']=m['source_fingerprint'];write(a.output/'release.json',release)
 with tarfile.open(a.artifacts/'detector-tiles-delta.tar.gz','w:gz') as t:
  for f in files+['manifest.json','config.json','release.json']:t.add(a.output/f,arcname=f)
 write(a.artifacts/'preparation-result.json',dict(added_tiles=len(extra),skipped_partial=skipped,dev_records_unchanged=True,counts=m['counts'],source_fingerprint=m['source_fingerprint'],manifest_sha256=sha(a.output/'manifest.json'),delta_sha256=sha(a.artifacts/'detector-tiles-delta.tar.gz'),elapsed=time.monotonic()-started))
if __name__=='__main__':main()
