"""Offline detector check: training-fit P/R and positive-only manual recall."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def iou(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def match(boxes,predictions,threshold):
    predictions = sorted((p for p in predictions if p['confidence'] >= threshold),
                         key=lambda p:p['confidence'],reverse=True)
    used, matches = set(), []
    for prediction in predictions:
        candidates = [(iou(prediction['xyxy'],box['xyxy']),i) for i,box in enumerate(boxes)
                      if i not in used and prediction['class_name']==box['class_name']]
        overlap,index = max(candidates, default=(0.,-1))
        if overlap >= .5:
            used.add(index)
            matches.append((index,prediction))
    return predictions,matches


def summarize(rows, threshold):
    totals = defaultdict(Counter)
    confusion = defaultdict(Counter)
    for row in rows:
        predictions,matches = match(row['boxes'],row['predictions'],threshold)
        matched = {i for i,_ in matches}
        subsets = [f"{row['subset']}/L{row['zoom']}"]
        for key in subsets:
            totals[key].update(views=1,gt=len(row['boxes']),tp=len(matches),predictions=len(predictions))
        for i,box in enumerate(row['boxes']):
            key = f"{row['subset']}/L{row['zoom']}/{box['class_name']}"
            nearby = sorted(((iou(box['xyxy'],p['xyxy']),p) for p in predictions),
                            key=lambda pair:pair[0], reverse=True)
            localized = bool(nearby and nearby[0][0] >= .5)
            totals[key].update(gt=1,tp=int(i in matched),localized=int(localized))
            totals[subsets[0]].update(localized=int(localized))
            if row['subset']=='manual_known_positives':
                predicted = box['class_name'] if i in matched else nearby[0][1]['class_name'] if localized else 'missed'
                confusion[f"L{row['zoom']}/{box['class_name']}"][predicted] += 1
        if row['subset']=='reference_training_fit':
            for pred in predictions:
                totals[f"{row['subset']}/L{row['zoom']}/{pred['class_name']}"].update(predictions=1)
    result = {}
    for key,value in sorted(totals.items()):
        row = dict(value)
        row['recall'] = value['tp']/value['gt'] if value['gt'] else None
        row['localization_recall'] = value['localized']/value['gt'] if value['gt'] else None
        if key.startswith('reference_training_fit/'):
            row['precision'] = value['tp']/value['predictions'] if value['predictions'] else None
        result[key] = row
    return dict(metrics=result, confusion=dict(confusion))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data',type=Path,required=True)
    parser.add_argument('--evaluation-data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    args = parser.parse_args()
    os.environ['WANDB_MODE']='disabled'
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2
    import torch
    from ultralytics import YOLO,settings
    from ultralytics.utils import LOGGER
    LOGGER.setLevel(logging.ERROR)
    settings.update({'wandb':False})
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    assert sha(args.checkpoint)==args.expected_checkpoint_sha256
    args.output.mkdir(parents=True,exist_ok=False)
    training = json.loads((args.training_data/'manifest.json').read_text())
    evaluation = json.loads((args.evaluation_data/'manifest.json').read_text())
    assert sha(args.training_data/'manifest.json')==evaluation['training_manifest_sha256']
    model = YOLO(str(args.checkpoint))
    assert [model.names[i] for i in range(len(model.names))]==training['classes']==evaluation['classes']
    views = []
    for row in training['records']:
        if row['task']!='detector':
            continue
        label_file = args.training_data/row['label_file']
        assert sha(label_file)==row['label_sha256']
        boxes = []
        for line in label_file.read_text().splitlines():
            c,x,y,w,h = map(float,line.split())
            boxes.append(dict(class_name=training['classes'][int(c)],
                              xyxy=[(x-w/2)*960,(y-h/2)*540,(x+w/2)*960,(y+h/2)*540]))
        views.append(dict(id=Path(row['file']).stem,subset='reference_training_fit',file=row['file'],
                          sha256=row['sha256'],zoom=row['zoom'],frame=row['frame'],boxes=boxes))
    views.extend(evaluation['records'])
    rows = []
    start = time.perf_counter()
    with (args.output/'predictions.jsonl').open('w') as stream:
        for index,row in enumerate(views):
            root = args.training_data if row['subset']=='reference_training_fit' else args.evaluation_data
            path = root/row['file']
            assert sha(path)==row['sha256']
            image = cv2.imread(str(path))
            result = model.predict(image,imgsz=960,device=0,half=True,conf=.01,max_det=300,verbose=False)[0]
            predictions = [dict(xyxy=box[:4],confidence=box[4],class_name=model.names[int(box[5])])
                           for box in result.boxes.data.cpu().tolist()]
            record = dict(row,predictions=predictions)
            stream.write(json.dumps(record)+'\n')
            rows.append(record)
            if (index+1)%50==0:
                print(json.dumps(dict(completed=index+1,total=len(views))),flush=True)
    report = dict(checkpoint_sha256=sha(args.checkpoint),evaluation_manifest_sha256=sha(args.evaluation_data/'manifest.json'),
                  script_sha256=sha(Path(__file__)),gpu=torch.cuda.get_device_name(0),views=len(rows),
                  elapsed_seconds=time.perf_counter()-start,match_iou=.5,primary_confidence=.25,
                  thresholds={str(t):summarize(rows,t) for t in (.1,.25,.5)},
                  limits=evaluation['limits']+['Reference metrics use the actual training images, not a holdout.',
                                               'Confidence thresholds were specified before predictions; this run does not tune or retrain.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report['thresholds']['0.25']['metrics'].items() if k.count('/')==1},indent=2))


if __name__=='__main__':
    main()
