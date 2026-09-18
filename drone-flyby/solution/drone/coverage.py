#!/usr/bin/env python3
"""Audit native 4x4 tile coverage without confusing partial canvases with originals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def audit(roots):
    grid=[(x,y,x+960,y+540) for y in range(0,2160,540) for x in range(0,3840,960)]
    frames=defaultdict(set)
    sources=defaultdict(list)
    for root in roots:
        for path in root.rglob('*.json'):
            record=json.loads(path.read_text())
            if 'view' not in record or record['view']['resolution_level']!=2:
                continue
            region=tuple(record['view']['source_region_xyxy'])
            if region not in grid:
                continue
            if not (path.parent/record['image_file']).is_file():
                continue
            frame=record['frame'];frames[frame].add(region);sources[region].append(frame)
    entries=[{'frame':frame,'complete':len(frames[frame])==16,'native_grid_tiles':len(frames[frame]),
              'missing_regions':[list(r) for r in grid if r not in frames[frame]]} for frame in range(1,250)]
    return {'complete_frames':sum(e['complete'] for e in entries),
            'complete_frame_numbers':[e['frame'] for e in entries if e['complete']],
            'tiles':[{'region':list(r),'unique_frames':len(set(sources[r]))} for r in grid],
            'frames':entries}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.roots)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
