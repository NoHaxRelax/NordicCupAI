"""Compare CNN, templates, and their union on identical, explicit cohorts."""
import argparse
import json
from pathlib import Path
from .evaluate import measure,nms


def summarize(cnn,features,concerns):
    lookup={(r['split'],r['frame']):r for r in features['results']}
    flagged={(r['frame'],r['track']) for r in concerns['concerns']}
    results=[]
    for threshold in [.01,.05,.1,.25,.5,.75]:
        for cohort in ['contained_boxes','unflagged_validation']:
            for split in ['reference','validation']:
                sums={engine:dict(matched=0,targets=0,proposals=0,localized=0) for engine in ['cnn','features','union']}
                for row in cnn['results']:
                    if row['split']!=split:continue
                    key=(split,row['frame'])
                    if key not in lookup:raise ValueError('Different frame cohorts')
                    truth=[t for t in row['truth'] if 0<t['bbox'][0]<t['bbox'][2]<3839 and 0<t['bbox'][1]<t['bbox'][3]<2159]
                    if cohort=='unflagged_validation' and split=='validation':
                        truth=[t for t in truth if (row['frame'],t.get('track')) not in flagged]
                    neural=[p for p in row['predictions'] if p['score']>=threshold]
                    deterministic=[p for p in lookup[key]['predictions'] if p['score']>=.8]
                    # Keep both model families' localizations: collapsing correlated
                    # different extents could hide a correct box. One-to-one matching
                    # and proposal counts penalize duplicate hypotheses explicitly.
                    families=dict(cnn=neural,features=deterministic,union=neural+deterministic)
                    for engine,predictions in families.items():
                        score=measure(predictions,truth);agnostic=measure(predictions,truth,False)
                        for k in ['matched','targets','proposals']:sums[engine][k]+=score[k]
                        sums[engine]['localized']+=agnostic['matched']
                results.append(dict(cnn_threshold=threshold,feature_threshold=.8,cohort=cohort,split=split,engines=sums))
    return dict(results=results,limitations=['Feature and CNN confidences are not calibrated to each other. Union retains both sets; duplicates count as proposals.',
            'Unflagged subset is a sensitivity analysis, not replacement ground truth.',
            'Localization counts ignore class; proposal counts are not false-positive counts on incomplete validation labels.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cnn',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--concerns',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=summarize(json.loads(a.cnn.read_text()),json.loads(a.features.read_text()),json.loads(a.concerns.read_text()))
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    for r in report['results']:
        if r['cnn_threshold']==.25:print(json.dumps(r))
