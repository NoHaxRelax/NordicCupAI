#!/usr/bin/env python3
"""Bounded live pilot: infer one label/box while acquiring the remaining grid tiles.

Requires explicit authorization for live validation runs and the active relay.
Never submits final evaluation. Maximum 30 validation submissions; normal path 20.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from oracle import (decode_rank, normalized_prediction, ranked_predictions,
                    edge_domain, scan_boxes, bracket_from_rank, fit_box, integer_candidates)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/drone-source-2026-09-17'))
from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame',type=int,default=140)
    parser.add_argument('--bbox',type=float,nargs=4,default=[715,53,825,137])
    parser.add_argument('--prefix',default='candidate140')
    args=parser.parse_args()
    if not args.prefix.replace('-','').isalnum() or len(args.prefix)>30:
        parser.error('Use a short alphanumeric prefix')
    seed=args.bbox
    remaining_tiles=[(x,y) for y in [270,810,1350,1890] for x in [480,1440,2400,3360] if (x,y)!=(480,270)]
    tiles=remaining_tiles+[(480,270)]+remaining_tiles
    history=[]
    folder=ROOT/'artifacts/drone-api-tests/score-probes'
    folder.mkdir(parents=True,exist_ok=True)

    def query(suffix,predictions):
        for row in predictions:DroneFlybyPredictionDto.model_validate(row)
        base_name=f'{args.prefix}-{len(history):02d}-{suffix}'
        for retry in range(4):
            name=base_name+(f'-retry{retry}' if retry else '')
            plan={'name':name,'target':list(tiles[len(history)]),'candidate_frame':args.frame,
                  'candidate_bbox_source':seed,'predictions_by_frame':{str(args.frame):predictions}}
            path=folder/f'{name}-plan.json'
            if not (folder/name/'queue.json').exists():
                submissions=len(list(folder.glob(f'{args.prefix}-*/queue.json')))
                if submissions>=30:
                    raise RuntimeError('Session limit reached, including retries; no further submissions')
                path.write_text(json.dumps(plan,indent=2)+'\n')
                subprocess.run([sys.executable,str(ROOT/'drone/run_probe.py'),str(path)],check=True)
            # Reuse only completed receipts. Never requeue an incomplete attempt.
            result=json.loads((folder/name/'result.json').read_text())
            if result.get('errors'):
                if (folder/name/'ignored-score.txt').exists():
                    print(json.dumps({'probe':name,'ignored':'manually reviewed transport failure','retry':retry}),flush=True)
                    continue
                raise RuntimeError('Probe had API errors; its score is not a clean signal')
            captures=list((ROOT/'data/drone/capture/mining'/name).rglob('*.json'))
            rows=[json.loads(p.read_text()) for p in captures]
            target=[r for r in rows if r['frame']==args.frame]
            if not target:
                (folder/name/'ignored-score.txt').write_text('Target frame was skipped. Score is not label/box evidence.\n')
                print(json.dumps({'probe':name,'ignored':'target frame skipped','retry':retry}),flush=True)
                continue
            if len(target)!=1 or target[0]['response']['annotations']!=predictions:
                raise RuntimeError('Target frame predictions not verified')
            history.append({'probe':name,'score':result['score'],'tile':plan['target'],'capture_count':len(captures)})
            (folder/f'{args.prefix}-history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(json.dumps({'probe':name,'score':result['score'],'recorded_requests':len(captures)}),flush=True)
            return result['score']
        raise RuntimeError('Target frame skipped on four attempts; stop and resolve transport latency')

    candidates=[normalized_prediction(seed,name,.5) for name in OBJECT_CLASSES]
    baseline=query('control',candidates)
    if baseline<=0:
        raise RuntimeError('Candidate box did not match any class; revise the visual proposal before more probes')
    coded=candidates+[normalized_prediction([1,1,2,2],name,.9)
                      for index,name in enumerate(OBJECT_CLASSES) for _ in range(index)]
    coded_score=query('label-rank',coded)
    label=OBJECT_CLASSES[decode_rank(baseline,coded_score,16)-1]
    confirm=query('label-confirm',[normalized_prediction(seed,label,1.0)])
    if abs(confirm-baseline)>max(1e-12,baseline*1e-7):
        raise RuntimeError('Single-class score does not reproduce the control; possible multiple-object ambiguity')
    brackets=[]
    unbounded=[]
    for edge in range(4):
        for side,start in enumerate(edge_domain(seed,edge)):
            end=seed[edge]
            for stage in range(2):
                boxes=scan_boxes(seed,edge,start,end)
                observed=query(f'edge{edge}-side{side}-stage{stage}',ranked_predictions(boxes,label))
                rank=decode_rank(baseline,observed)
                if rank == 1:
                    # A source border can leave no valid negative candidate on this
                    # side. Preserve the observed positive bound, then fit from the
                    # remaining nondegenerate boundary scans.
                    unbounded.append({'edge':edge,'side':side,'stage':stage,
                                      'positive_box':boxes[0],'rank':rank,
                                      'reason':'first candidate already matches'})
                    (folder/f'{args.prefix}-unbounded.json').write_text(json.dumps(unbounded,indent=2)+'\n')
                    bracket=None
                    break
                bracket=bracket_from_rank(boxes,rank,edge)
                start=bracket['negative_box'][edge];end=bracket['positive_box'][edge]
            if bracket is None:
                continue
            bracket['side']=side
            brackets.append(bracket)
            (folder/f'{args.prefix}-brackets.json').write_text(json.dumps(brackets,indent=2)+'\n')
    if len(brackets)<4 or len({b['edge'] for b in brackets})<3:
        raise RuntimeError('Too few nondegenerate boundary scans to fit one box')
    fit=fit_box(seed,brackets)
    if not fit['success'] or fit['residual_norm']>.01:
        raise RuntimeError('Boundary measurements do not fit one clean box; inspect ambiguity before continuing')
    integers=integer_candidates(fit['bbox'],brackets)
    final_box=integers[0] if len(integers)==1 else fit['bbox']
    final_score=query('box-confirm',[normalized_prediction(final_box,label,1.0)])
    if abs(final_score-baseline)>max(1e-12,baseline*1e-7):
        raise RuntimeError('Recovered box did not reproduce the isolated match score')
    output={'frame':args.frame,'object_id':label,'bbox_source_xyxy':final_box,
            'bbox_global_xyxy':normalized_prediction(final_box,label,1.0)['bbox'],
            'provenance':'inferred through isolated validation score probes; not an organizer label export',
            'label_status':'rank_decoded_and_single_class_confirmed',
            'box_status':'threshold_constrained_and_match_confirmed',
            'integer_assumption':'integer coordinates, as in the public reference annotations',
            'nearby_integer_candidates':integers,'fit':fit,'brackets':brackets,'probes':history}
    output['unbounded_scans']=unbounded
    destination=ROOT/'data/drone/mined';destination.mkdir(parents=True,exist_ok=True)
    (destination/f'{args.prefix}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'mined_label':label,'source_box':final_box,'nearby_integer_candidates':len(integers),
                      'runs':len(history),'output':str(destination/f'{args.prefix}.json')}),flush=True)


if __name__=='__main__':main()
