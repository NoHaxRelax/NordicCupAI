"""Freeze matched 384/256 source-square experiments inside approved parent regions."""
import argparse, copy, json, hashlib
from pathlib import Path
import cv2
from drone.grid_dataset.build import ROOT, digest, write, intersection, contains

def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'data/drone/grid384-20260918-v1'
    m=json.loads((base/'manifest-combined-approved.json').read_text())
    bytile={}
    for r in m['records']:
        if r['split'] in ('train','dev'):bytile.setdefault(r['tile_id'],{})[r['zoom']]=r
    for size in [384,256]:
        dest=output/str(size);dest.mkdir();rows=[];coverage=[]
        for tid,zooms in sorted(bytile.items()):
            original=zooms[2];native=cv2.imread(str(base/original['file']))
            assert digest(base/original['file'])==original['sha256']
            # A centred nested grid gives identical crop placement at all zooms.
            windows=[[0,0,384,384]] if size==384 else [[x,y,x+256,y+256] for y in [0,64,128] for x in [0,64,128]]
            scored=[]
            for box in windows:
                anns=[]
                for a in original['annotations']:
                    hit=intersection(box,a['bbox_xyxy'])
                    if hit:
                        aa=copy.deepcopy(a);aa['bbox_xyxy']=[hit[0]-box[0],hit[1]-box[1],hit[2]-box[0],hit[3]-box[1]]
                        aa['fully_contained']=a['fully_contained'] and contains(box,a['bbox_xyxy']);anns.append(aa)
                full=sum(a['fully_contained'] for a in anns)
                # Preserve known positive identity; never turn a positive tile into
                # negative on the basis of an incomplete annotation set.
                if original['kind']=='positive' and not full:continue
                scored.append((full,box,anns))
            if not scored:
                coverage.append(dict(tile_id=tid,reason='No 256 square fully contains any original target',split=original['split']))
                continue
            # One window per original tile, maximizing retained targets; ties stable.
            scored.sort(key=lambda t:(-t[0],hashlib.sha256((tid+str(t[1])).encode()).hexdigest()))
            _,box,anns=scored[0];x,y,u,v=box
            for z in [0,1,2]:
                rec=copy.deepcopy(zooms[z]);divisor=[4,2,1][z]
                low=cv2.resize(native[y:v,x:u],(size//divisor,size//divisor),interpolation=cv2.INTER_AREA)
                image=cv2.resize(low,(size,size),interpolation=cv2.INTER_LINEAR)
                path=dest/'images'/f'{tid}-L{z}.png';path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(path),image)
                rec.update(file=str(path.relative_to(dest)),sha256=digest(path),input_size=size,native_crop_size=size//divisor,annotations=anns,parent_tile_id=tid,parent_manifest_sha256=digest(base/'manifest-combined-approved.json'))
                sx,sy,_,_=original['source_rect_xyxy'];rec['source_rect_xyxy']=[sx+x,sy+y,sx+u,sy+v]
                rows.append(rec)
        write(dest/'manifest.json',dict(classes=m['classes'],records=rows,source_manifest_sha256=digest(base/'manifest-combined-approved.json'),excluded=coverage,notes=['Nested crop comparison; legal full-image scanning requires separate evaluation.','Validation class negatives are masked; this is recognition, not detector training.']))
    a=json.loads((output/'384/manifest.json').read_text());b=json.loads((output/'256/manifest.json').read_text())
    common={r['id'] for r in a['records']} & {r['id'] for r in b['records']}
    # Fair paired comparison uses precisely the common parent IDs for both sizes.
    for size in [384,256]:
        p=output/str(size)/'manifest.json';d=json.loads(p.read_text());d['records']=[r for r in d['records'] if r['id'] in common];write(p,d)
        assert {a['class_name'] for r in d['records'] if r['split']=='train' for a in r['annotations'] if a['fully_contained']}==set(m['classes'])
        print(size,len(d['records']), 'images',len(d['excluded']),'excluded parent squares')
    write(output/'comparison.json',dict(common_images=len(common),source_manifest_sha256=digest(base/'manifest-combined-approved.json'),sizes=[256,384]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
