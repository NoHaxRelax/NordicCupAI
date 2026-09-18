"""Mine false detections only from explicitly empty TRAIN images.

Writes a new immutable dataset with bounded repeats of difficult empty images.
Never reads evaluation images or turns incomplete validation regions negative.
"""
import argparse
import json
from pathlib import Path
import shutil
from .prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--limit',type=int,default=64)
    a=p.parse_args()
    from ultralytics import YOLO
    manifest=json.loads((a.data/'manifest.json').read_text())
    candidates=[r for r in manifest['records'] if r['split']=='train' and not r['objects']]
    model=YOLO(str(a.weights));scores=[]
    for row in candidates:
        image=a.data/row['file'];label=a.data/row['label']
        if sha(image)!=row['sha256'] or sha(label)!=row['label_sha256'] or label.read_text().strip():raise ValueError('Invalid empty training image')
        prediction=model.predict(str(image),imgsz=960,conf=.1,device=a.device,verbose=False)[0]
        if len(prediction.boxes):scores.append(dict(record=row,score=float(prediction.boxes.conf.max().cpu()),detections=len(prediction.boxes)))
    scores.sort(key=lambda r:(-r['score'],r['record']['file']))
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ['images','labels']:shutil.copytree(a.data/name,a.output/name)
    records=manifest['records'].copy()
    for index,row in enumerate(scores[:a.limit]):
        original=row['record']
        for repeat in range(3):
            name=f'hard-{index:04d}-{repeat}'
            dest=a.output/'images/train'/f'{name}.jpg';label=a.output/'labels/train'/f'{name}.txt'
            shutil.copyfile(a.data/original['file'],dest);label.write_text('')
            records.append(dict(split='train',file=str(dest.relative_to(a.output)),sha256=sha(dest),label=str(label.relative_to(a.output)),
                                label_sha256=sha(label),objects=[],mined_from=original['file'],mining_score=row['score']))
    manifest.update(records=records,base_manifest_sha256=sha(a.data/'manifest.json'),hard_negative_mining=dict(checkpoint_sha256=sha(a.weights),
                    empty_training_images=len(candidates),false_positive_images=len(scores),selected=scores[:a.limit],repeats=3))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(empty_training_images=len(candidates),false_positive_images=len(scores),selected=min(a.limit,len(scores)))))


if __name__=='__main__':main()
