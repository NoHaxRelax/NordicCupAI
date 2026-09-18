"""Score-probe geometry and decoding. Coordinates here are source pixels."""
import itertools
import math
import numpy as np
from scipy.optimize import least_squares


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def decode_rank(reference_score, observed_score, max_rank=100):
    if observed_score <= 0:
        return None
    ratio = reference_score / observed_score
    rank = round(ratio)
    if not 1 <= rank <= max_rank or not math.isclose(ratio, rank, abs_tol=1e-6, rel_tol=1e-6):
        raise ValueError(f'Score ratio {ratio} is not an isolated first-match rank')
    return rank


def edge_domain(box, edge):
    return [(0.0, box[2]-.01), (0.0, box[3]-.01),
            (box[0]+.01, 3840.0), (box[1]+.01, 2160.0)][edge]


def scan_boxes(seed, edge, negative_end, positive_end, count=100):
    result = []
    for value in np.linspace(negative_end, positive_end, count):
        box = list(seed)
        box[edge] = float(value)
        result.append(box)
    return result


def bracket_from_rank(boxes, rank, edge):
    if rank is None:
        raise ValueError('The positive anchor failed; cannot decode the scan')
    if rank == 1:
        raise ValueError('First candidate already matches; no negative boundary bracket')
    return {'edge': edge, 'negative_box': boxes[rank-2], 'positive_box': boxes[rank-1], 'rank': rank}


def normalized_prediction(box, label, confidence):
    return {'object_id': label, 'bbox': [v/d for v,d in zip(box,[3840,2160,3840,2160])],
            'confidence': confidence}


def ranked_predictions(boxes, label):
    return [normalized_prediction(box, label, (len(boxes)-i)/(len(boxes)+1)) for i,box in enumerate(boxes)]


def fit_box(seed, brackets):
    midpoints = [(np.array(b['negative_box'])+np.array(b['positive_box']))/2 for b in brackets]
    def unpack(params):
        cx, cy, lw, lh = params
        w,h=np.exp(lw),np.exp(lh)
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
    initial = [(seed[0]+seed[2])/2,(seed[1]+seed[3])/2,np.log(seed[2]-seed[0]),np.log(seed[3]-seed[1])]
    def residual(params):
        box=unpack(params)
        return [iou(p,box)-.5 for p in midpoints]
    # IoU is piecewise smooth; one initialization can stop at a bad kink.
    starts=[]
    for sx,sy in itertools.product((.6,.8,1.,1.2,1.6),repeat=2):
        starts.append([initial[0],initial[1],initial[2]+math.log(sx),initial[3]+math.log(sy)])
    results=[least_squares(residual,start,max_nfev=1000,xtol=1e-11,ftol=1e-11,gtol=1e-11) for start in starts]
    result=min(results,key=lambda r:np.linalg.norm(r.fun))
    return {'bbox': unpack(result.x), 'residual_norm': float(np.linalg.norm(result.fun)), 'success': bool(result.success)}


def integer_candidates(fit, brackets, radius=2):
    """Nearby integer candidates consistent with every positive/negative result.

    This is a local enumeration, not a proof that no distant solutions exist.
    """
    axes=[range(round(v)-radius,round(v)+radius+1) for v in fit]
    result=[]
    for box in itertools.product(*axes):
        if not (0 <= box[0]<box[2]<=3840 and 0<=box[1]<box[3]<=2160):
            continue
        if all(iou(b['positive_box'],box)>=.5-1e-12 and iou(b['negative_box'],box)<.5 for b in brackets):
            result.append(list(box))
    return result
