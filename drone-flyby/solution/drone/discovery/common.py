"""Small shared helpers for the bounded, non-LLM discovery job."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/drone/discovery'
PRIVATE = Path('/private/tmp/nordic-ai-cup-secrets')
W, H = 3840, 2160

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, separators=(',', ':'))+'\n')
    temporary.replace(path)

def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def iou(a, b):
    n = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return n / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-n)

def hypothesis(frame, label, box):
    b = [round(float(v),3) for v in box]
    if not (0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H):
        return None
    return {'frame':int(frame),'class':label,'bbox_source_xyxy':b}

def predictions(items):
    frames = {}
    counts = {}
    for p in items:
        frame, label = str(p['frame']), p['class']
        b = p['bbox_source_xyxy']
        assert hypothesis(p['frame'],label,b) is not None
        key = (frame,label)
        counts[key] = counts.get(key,0)+1
        assert counts[key] <= 100, 'COCO class/frame cap exceeded'
        frames.setdefault(frame,[]).append({'object_id':label,
            'bbox':[b[0]/W,b[1]/H,b[2]/W,b[3]/H],
            'confidence':1-counts[key]/102})
        assert len(frames[frame]) <= 500, 'Response prediction cap exceeded'
    return frames

def split(items):
    """Keep frames/classes together where possible to make localization useful."""
    for field in ('frame','class'):
        values = sorted({x[field] for x in items})
        if len(values)>1:
            left = set(values[:len(values)//2])
            return [x for x in items if x[field] in left], [x for x in items if x[field] not in left]
    ordered = sorted(items, key=lambda p: (p['bbox_source_xyxy'][0]+p['bbox_source_xyxy'][2],p['bbox_source_xyxy'][1]+p['bbox_source_xyxy'][3]))
    middle = len(ordered)//2
    return ordered[:middle],ordered[middle:]
