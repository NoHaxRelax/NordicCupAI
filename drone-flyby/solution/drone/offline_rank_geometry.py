#!/usr/bin/env python3
"""Prove rank label coding and threshold geometry using the official local scorer."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from oracle import decode_rank, edge_domain, scan_boxes, bracket_from_rank, fit_box, integer_candidates

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/drone-source-2026-09-17'))
import local_evaluator as scorer
from dtos import OBJECT_CLASSES


def main():
    annotations={int(p.stem.split('_')[-1]):json.loads(p.read_text())['annotations']
                 for p in (ROOT/'data/drone/reference/helsinki/annotations').glob('*.json')}
    scorer.frame_numbers=lambda _:sorted(annotations)
    scorer.load_annotations=lambda frame,_:annotations[frame]
    def score(frame, predictions):
        with redirect_stdout(io.StringIO()):
            return scorer.score('reference',{frame:predictions})[0]
    labels=[]
    for expected in OBJECT_CLASSES:
        frame,target=next((frame,a) for frame,rows in annotations.items() for a in rows if a['object_id']==expected)
        base=[{'object_id':name,'bbox':target['bbox'],'confidence':.5} for name in OBJECT_CLASSES]
        coded=base+[{'object_id':name,'bbox':[1,1,2,2],'confidence':.9} for i,name in enumerate(OBJECT_CLASSES) for _ in range(i)]
        a,b=score(frame,base),score(frame,coded)
        inferred=OBJECT_CLASSES[decode_rank(a,b,16)-1]
        assert inferred==expected
        labels.append({'expected':expected,'inferred':inferred,'baseline':a,'coded':b})
    target=annotations[0][0]
    true_box=target['bbox']
    seed=[true_box[0]-3,true_box[1]-4,true_box[2]+5,true_box[3]+2]
    label=target['object_id']
    baseline=score(0,[{'object_id':label,'bbox':seed,'confidence':1.0}])
    brackets=[]
    for edge in range(4):
        for side,start in enumerate(edge_domain(seed,edge)):
            end=seed[edge]
            for stage in range(2):
                boxes=scan_boxes(seed,edge,start,end)
                predictions=[{'object_id':label,'bbox':b,'confidence':(100-i)/101} for i,b in enumerate(boxes)]
                s=score(0,predictions)
                bracket=bracket_from_rank(boxes,decode_rank(baseline,s),edge)
                start=bracket['negative_box'][edge];end=bracket['positive_box'][edge]
            bracket['side']=side
            brackets.append(bracket)
    fitted=fit_box(seed,brackets)
    candidates=integer_candidates(fitted['bbox'],brackets)
    result={'dataset':'public Helsinki reference, not validation','live_queries':0,
            'label_rank_checks':labels,'geometry':{'seed':seed,'true_box':true_box,'fit':fitted,
            'integer_candidates':candidates,'brackets':brackets,'queries':16}}
    output=ROOT/'artifacts/drone-api-tests/offline-rank-geometry.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'label_rank_checks_passed':len(labels),'geometry':result['geometry']},indent=2))
    assert true_box in candidates


if __name__=='__main__':main()
