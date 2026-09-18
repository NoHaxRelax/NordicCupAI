#!/usr/bin/env python3
"""Build native-detail tiled sheets for deliberate full-frame visual review."""
import argparse
from pathlib import Path
import cv2

def main():
 p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 # Each output holds one spatial quadrant across 16 frames.  At 960x540 per cell,
 # small objects remain inspectable instead of being lost in full-frame thumbnails.
 frames=list(range(a.start,a.end+1)); cols=4
 for q,(x,y) in enumerate(((0,0),(1920,0),(0,1080),(1920,1080))):
  cells=[]
  for f in frames:
   im=cv2.imread(str(a.images/f'frame_{f:06d}.png'),cv2.IMREAD_UNCHANGED)
   tile=im[y:y+1080,x:x+1920,:3]
   tile=cv2.resize(tile,(960,540),interpolation=cv2.INTER_AREA)
   cv2.putText(tile,str(f),(15,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3,cv2.LINE_AA);cells.append(tile)
  while len(cells)%cols: cells.append(cells[-1]*0)
  rows=[cv2.hconcat(cells[i:i+cols]) for i in range(0,len(cells),cols)]
  cv2.imwrite(str(a.output/f'frames_{a.start:03d}_{a.end:03d}_q{q}.jpg'),cv2.vconcat(rows),[cv2.IMWRITE_JPEG_QUALITY,92])
if __name__=='__main__':main()
