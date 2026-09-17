"""Render fixed representative targets at three zooms and full L1 views."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_detector import iou


GREEN = (70,210,90)
MAGENTA = (220,90,235)


def label(image,text,origin,color=(235,235,235),size=.5):
    cv2.putText(image,text,origin,cv2.FONT_HERSHEY_SIMPLEX,size,(20,20,20),3,cv2.LINE_AA)
    cv2.putText(image,text,origin,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


def overlay(image,row):
    image = image.copy()
    for prediction in row['predictions']:
        if prediction['confidence'] < .25:
            continue
        x1,y1,x2,y2 = map(round,prediction['xyxy'])
        cv2.rectangle(image,(x1,y1),(x2,y2),MAGENTA,2)
        label(image,f"{prediction['class_name']} {prediction['confidence']:.2f}",(max(0,x1),max(15,y1-5)),MAGENTA)
    for box in row['boxes']:
        x1,y1,x2,y2 = map(round,box['xyxy'])
        cv2.rectangle(image,(x1,y1),(x2,y2),GREEN,1)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--evaluation-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    rows = [r for r in rows if r['subset']=='manual_known_positives']
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    selected = []
    for group,group_rows in sorted(groups.items()):
        frames = sorted({r['frame'] for r in group_rows})
        # Fixed median frame per track, chosen without inspecting predictions.
        frame = frames[len(frames)//2]
        selected.extend(sorted((r for r in group_rows if r['frame']==frame),key=lambda r:r['zoom']))
    sheet = np.full((110+len(groups)*235,1440,3),24,dtype=np.uint8)
    label(sheet,'Trained YOLO26x: six fixed representative validation targets',(20,30),size=.8)
    label(sheet,'Green: manual box | Purple: prediction >= 0.25 | Closeups enlarged to show pixels',(20,58),size=.6)
    for zoom in range(3):
        label(sheet,f'L{zoom}: source downsampled {4//2**zoom}x',(20+480*zoom,90),size=.65)
    manifest=[]
    for index,row in enumerate(selected):
        image=cv2.imread(str(args.evaluation_data/row['file']))
        box=row['boxes'][0]
        candidates=sorted((p for p in row['predictions'] if p['confidence']>=.25),
                          key=lambda p:iou(box['xyxy'],p['xyxy']),reverse=True)
        correct=[p for p in candidates if p['class_name']==box['class_name'] and iou(box['xyxy'],p['xyxy'])>=.5]
        nearby=candidates[0] if candidates and iou(box['xyxy'],candidates[0]['xyxy'])>=.5 else None
        state='CORRECT' if correct else 'WRONG CLASS' if nearby else 'MISSED'
        title=f"{box['class_name']} f{row['frame']} | {state}"
        x1,y1,x2,y2=box['xyxy']
        cx,cy=(x1+x2)/2,(y1+y2)/2
        w,h=max(130,x2-x1+60),max(80,y2-y1+45)
        left,top=max(0,round(cx-w/2)),max(0,round(cy-h/2))
        right,bottom=min(960,round(cx+w/2)),min(540,round(cy+h/2))
        # Boxes drawn before enlargement retain the real pixel geometry.
        patch=overlay(image,row)[top:bottom,left:right]
        factor=min(450/patch.shape[1],160/patch.shape[0])
        patch=cv2.resize(patch,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
        ox,oy=480*row['zoom'],110+235*(index//3)
        label(sheet,title,(ox+10,oy+20),GREEN if correct else (80,130,255),.53)
        label(sheet,f"Object: {x2-x1:.0f} x {y2-y1:.0f} pixels",(ox+10,oy+43),size=.48)
        px,py=ox+10,oy+54
        sheet[py:py+patch.shape[0],px:px+patch.shape[1]]=patch
        manifest.append(dict(id=row['id'],class_name=box['class_name'],zoom=row['zoom'],frame=row['frame'],status=state))
    cv2.imwrite(str(args.output/'zoom-comparison.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
    full=np.full((90+3*575,1920,3),24,dtype=np.uint8)
    label(full,'L1 camera views: all predictions >= 0.25; green boxes are known labels only',(20,32),size=.8)
    label(full,'Other predictions may be real unannotated objects. These frames are not exhaustively labeled.',(20,62),size=.65)
    for index,row in enumerate(r for r in selected if r['zoom']==1):
        image=overlay(cv2.imread(str(args.evaluation_data/row['file'])),row)
        ox,oy=(index%2)*960,90+(index//2)*575
        label(full,f"{row['boxes'][0]['class_name']} / frame {row['frame']}",(ox+10,oy+23),size=.65)
        full[oy+30:oy+570,ox:ox+960]=image
    cv2.imwrite(str(args.output/'l1-full-views.jpg'),full,[cv2.IMWRITE_JPEG_QUALITY,94])
    (args.output/'representative-cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
