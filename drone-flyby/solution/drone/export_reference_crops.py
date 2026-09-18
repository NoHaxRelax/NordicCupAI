"""Export labelled reference object crops using the official zoom rendering.

This is reference data, not recovered validation ground truth. Crops retain
background pixels; these are not transparent assets from the simulator.
"""
import json
import math
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/drone/reference/helsinki'
OUT = ROOT / 'data/drone/training/reference-crops'
W, H = 3840, 2160

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    counts = Counter()
    for path in sorted((SOURCE / 'annotations').glob('*.json')):
        document = json.loads(path.read_text())
        frame = document['frame']
        source = cv2.imread(str(SOURCE / 'images' / path.with_suffix('.png').name))
        assert source is not None and source.shape[:2] == (H,W)
        overview = cv2.resize(source, (960,540), interpolation=cv2.INTER_AREA)
        for index, annotation in enumerate(document['annotations']):
            label = annotation['object_id']
            box = annotation['bbox']
            cx,cy = round((box[0]+box[2])/2),round((box[1]+box[3])/2)
            counts[label] += 1
            for level, (rw,rh) in enumerate(((3840,2160),(1920,1080),(960,540))):
                if level == 0:
                    x,y=0,0
                    view=overview
                else:
                    x=max(0,min(W-rw,cx-rw//2))
                    y=max(0,min(H-rh,cy-rh//2))
                    region=source[y:y+rh,x:x+rw]
                    view=cv2.resize(region,(960,540),interpolation=cv2.INTER_AREA) if level == 1 else region
                scale=960/rw
                vb=[(box[0]-x)*scale,(box[1]-y)*scale,(box[2]-x)*scale,(box[3]-y)*scale]
                # A small contextual border avoids cutting the object at a rounded edge.
                crop_box=[max(0,math.floor(vb[0])-8),max(0,math.floor(vb[1])-8),min(960,math.ceil(vb[2])+8),min(540,math.ceil(vb[3])+8)]
                a,b,c,d=crop_box
                folder=OUT/label/f'L{level}'
                folder.mkdir(parents=True,exist_ok=True)
                filename=folder/f'frame_{frame:06d}_object_{index:02d}.png'
                assert cv2.imwrite(str(filename),view[b:d,a:c])
                records.append({'split':'public_reference','class':label,'frame':frame,'zoom_level':level,
                    'reference_instance_id':f'helsinki-{label}',
                    'crop_file':str(filename.relative_to(OUT)),
                    'bbox_source_xyxy':box,'bbox_normalized_xyxy':[box[0]/W,box[1]/H,box[2]/W,box[3]/H],
                    'bbox_view_xyxy':vb,'bbox_crop_xyxy':[vb[0]-a,vb[1]-b,vb[2]-a,vb[3]-b],
                    'crop_region_in_view_xyxy':crop_box,'source_region_xyxy':[x,y,x+rw,y+rh],
                    'pose':document.get('pose'), 'source_edge_clipped':box[0]<=0 or box[1]<=0 or box[2]>=W-1 or box[3]>=H-1,
                    'annotation_evidence':'organizer_public_reference','rendering':'official INTER_AREA full-view resize before object crop'})
    manifest={'description':'Ground-truth-labelled public reference crops at three zoom levels, retaining image background.',
              'independent_instances':16,'source_frames':25,'source_annotations':sum(counts.values()),
              'zoom_crops':len(records),'class_annotation_counts':dict(sorted(counts.items())),
              'limitations':['Views of one physical reference instance per class, not independent samples.',
                             'Does not establish complete validation or evaluation appearance distribution.',
                             'No simulator sprite alpha masks; context/background remains.',
                             'Crop regions reproduce individual views; they are not a reachable sequential camera policy.'],
              'records':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(records)==sum(counts.values())*3
    assert len(counts)==16
    # Show representative labelled crops for every class and zoom.
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(1100,16*230+55),'#101b27'); draw=ImageDraw.Draw(sheet)
    for j,title in enumerate(('Class / reference views','L0 (quarter scale)','L1 (half scale)','L2 (native)')):
        draw.text((15+j*275,15),title,fill='white')
    for i,label in enumerate(sorted(counts)):
        candidates=[r for r in records if r['class']==label and r['zoom_level']==2 and not r['source_edge_clipped']]
        best=max(candidates or [r for r in records if r['class']==label and r['zoom_level']==2],key=lambda r:(r['bbox_source_xyxy'][2]-r['bbox_source_xyxy'][0])*(r['bbox_source_xyxy'][3]-r['bbox_source_xyxy'][1]))
        yy=55+i*230
        draw.text((15,yy+25),f'{label} ({counts[label]} views)',fill='white')
        for level in range(3):
            r=next(r for r in records if r['class']==label and r['frame']==best['frame'] and r['zoom_level']==level)
            with Image.open(OUT/r['crop_file']) as im:
                sheet.paste(im,(275*(level+1)+10,yy))
    sheet.save(OUT/'class-zoom-atlas.png')
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2))

if __name__=='__main__':main()
