"""Bounded offline grid-probe experiments; no API calls or hidden labels."""
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/drone-source-2026-09-17'))
import local_evaluator as scorer
from faster_coco_eval.core.cocoeval import Params

def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

def score(truth, guesses):
    scorer.frame_numbers=lambda _: [0]
    scorer.load_annotations=lambda frame,scene:[{'object_id':'helicopter','bbox':b} for b in truth]
    predictions=[{'object_id':'helicopter','bbox':b,'confidence':1-i/(len(guesses)+1)} for i,b in enumerate(guesses)]
    with redirect_stdout(io.StringIO()):
        return scorer.score('synthetic-grid-proof',{0:predictions})[0]

def main():
    truth=[50,50,150,150]
    tiled=[[x,y,x+100,y+100] for y in (0,100) for x in (0,100)]
    shifted=[[x,y,x+100,y+100] for y in range(0,101,25) for x in range(0,101,25)]
    fp=[500,500,501,501]
    other=[250,250,350,350]
    out={'live_queries':0,'ground_truth':'synthetic, not validation',
         'nonoverlapping_max_iou':max(iou(truth,b) for b in tiled),
         'nonoverlapping_score':score([truth],tiled),
         'overlapping_grid_score':score([truth],shifted),
         'two_exact_boxes_score':score([truth,other],[truth,other]),
         'duplicate_before_second_match_score':score([truth,other],[truth,truth,other]),
         'match_at_rank100_score':score([truth],[fp]*99+[truth]),
         'match_at_rank101_score':score([truth],[fp]*100+[truth]),
         'default_max_detections':Params('bbox').maxDets,
         'same_size_quarter_stride_worst_iou':49/79,
         '100px_square_grid_candidates':(math.ceil((3840-100)/25)+1)*(math.ceil((2160-100)/25)+1)}
    assert out['nonoverlapping_score']==0 and out['overlapping_grid_score']>0
    assert out['two_exact_boxes_score']>out['duplicate_before_second_match_score']>0
    assert out['match_at_rank100_score']>0 and out['match_at_rank101_score']==0
    path=ROOT/'artifacts/drone-api-tests/offline-grid-proof.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
