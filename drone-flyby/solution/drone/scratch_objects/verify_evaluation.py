"""Audit proposal-verifier tradeoffs without changing the raw detector report."""
import argparse,json
from pathlib import Path
from .evaluate import measure
from .prepare import sha


def filter_predictions(rows,threshold,mode,verifier_threshold):
    output=[]
    for p in rows:
        if p['score']<threshold:continue
        if mode=='class_gate' and p['verifier_probability']<verifier_threshold:continue
        if mode=='background_gate' and p['background_probability']>=verifier_threshold:continue
        if mode=='reclassify':
            if p['verifier_class']=='background' or p['verifier_top_probability']<verifier_threshold:continue
            p={**p,'class':p['verifier_class'],'original_class':p['class']}
        output.append(p)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=json.loads(a.report.read_text());features=json.loads(a.features.read_text())
    lookup={(r['split'],r['frame']):r['predictions'] for r in features['results']}
    rows=[]
    for mode,thresholds in [('raw',[0]),('class_gate',[.25,.5,.75]),('background_gate',[.5]),('reclassify',[.5,.75])]:
        for verifier_threshold in thresholds:
            for cnn_threshold in [.1,.25,.5,.75]:
                for split in ['reference','validation']:
                    result={k:dict(matched=0,targets=0,proposals=0,localized=0) for k in ['cnn','union']}
                    for image in report['results']:
                        if image['split']!=split:continue
                        truth=[t for t in image['truth'] if 0<t['bbox'][0]<t['bbox'][2]<3839 and 0<t['bbox'][1]<t['bbox'][3]<2159]
                        filtered=filter_predictions(image['predictions'],cnn_threshold,mode,verifier_threshold)
                        for name,proposals in [('cnn',filtered),('union',filtered+lookup[(split,image['frame'])])]:
                            score=measure(proposals,truth)
                            for k in ['matched','targets','proposals']:result[name][k]+=score[k]
                            result[name]['localized']+=measure(proposals,truth,False)['matched']
                    rows.append(dict(mode=mode,cnn_threshold=cnn_threshold,verifier_threshold=verifier_threshold,split=split,engines=result))
    output=dict(reranked_report_sha256=sha(a.report),feature_report_sha256=sha(a.features),results=rows,
                limitations=['Development sweep, not untouched test performance.','Union retains duplicate hypotheses and counts every proposal.',
                             'Reclassification retains detector score as an uncalibrated proposal ranking; not a probability for the new class.'])
    with a.output.open('x') as f:json.dump(output,f,indent=2)
    for r in rows:
        if r['cnn_threshold']==.25 and r['verifier_threshold'] in [0,.5]:print(json.dumps(r))


if __name__=='__main__':main()
