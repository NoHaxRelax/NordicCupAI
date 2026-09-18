"""Compare a frozen deterministic feature matcher with saved blind YOLO proposals."""
import argparse
from pathlib import Path
import time

from common import read, sha, write
from evaluate_views import metrics, iou


def merge(detector, templates):
    # Preserve learned localizations where both engines agree. Similarity scores
    # and neural confidences are not calibrated to one common scale.
    return detector + [p for p in templates if not any(
        p['class']==d['class'] and iou(p['bbox'],d['bbox'])>=.5 for d in detector)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--learned',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Use a new output directory')
    import cv2
    from template_matching.features import FeatureEnsemble,EnsembleSettings
    cv2.setNumThreads(4)
    manifest=read(a.data/'manifest.json');learned=read(a.learned)
    if learned['manifest_sha256']!=sha(a.data/'manifest.json'):raise ValueError('Learned predictions use another dataset')
    bank=read(a.bank/'manifest.json')
    if bank.get('validation_train_through',0)>=manifest['config']['validation_dev_frames'][0]:raise ValueError('Bank overlaps development')
    engine=FeatureEnsemble(a.bank,EnsembleSettings(score_threshold=.75,matching='global'))
    records={r['file']:r for r in manifest['records'] if r['task']=='detector' and r['split']=='dev'}
    rows=[];a.output.mkdir(parents=True);start=time.monotonic()
    for index,row in enumerate(learned['results']):
        record=records[row['file']];path=a.data/row['file']
        if sha(path)!=record['sha256']:raise ValueError('Changed view')
        image=cv2.imread(str(path))
        predictions=engine.detect(image,1/(4,2,1)[row['zoom']])
        rows.append(dict(row,learned=row['predictions'],predictions=predictions))
        write(a.output/'progress.json',dict(completed=index+1,total=len(learned['results']),elapsed=time.monotonic()-start))
    write(a.output/'predictions.json',dict(manifest_sha256=sha(a.data/'manifest.json'),bank_sha256=sha(a.bank/'manifest.json'),results=rows))
    reports={}
    for det_threshold in [.05,.1,.25,.5]:
        for template_threshold in [.75,.8,.85,.9]:
            cohorts={'learned':[],'deterministic':[],'hybrid':[]}
            for row in rows:
                detector=[p for p in row['learned'] if p['score']>=det_threshold]
                templates=[p for p in row['predictions'] if p['score']>=template_threshold]
                for name,preds in [('learned',detector),('deterministic',templates),('hybrid',merge(detector,templates))]:
                    cohorts[name].append(dict(row,predictions=preds))
            reports[f'det{det_threshold}-template{template_threshold}']={name:{f'L{z}':metrics([r for r in selected if r['zoom']==z],manifest['classes'],0)
                for z in range(3)} for name,selected in cohorts.items()}
    write(a.output/'summary.json',dict(grid=reports,elapsed_seconds=time.monotonic()-start,
        bank_sha256=sha(a.bank/'manifest.json'),learned_sha256=sha(a.learned),
        limitations=manifest['limitations']+['Development threshold grid; engine scores are not calibrated. Learned localization takes priority for agreeing same-class boxes.']))
    print('Hybrid comparison completed',flush=True)


if __name__=='__main__':main()
