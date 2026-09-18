#!/usr/bin/env python3
"""Render a local contact sheet and self-contained overlay viewer for the candidate pass."""
import argparse,json
from pathlib import Path
import cv2

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.data.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True)
 chosen=[5,66,127,140,146,188,249]; cells=[]
 for f in chosen:
  im=cv2.imread(str(a.images/f'frame_{f:06d}.png'),cv2.IMREAD_COLOR)
  for x in d['annotations']:
   if x['frame']!=f: continue
   b=list(map(int,x['bbox_source_xyxy'])); color=(0,220,0) if not x['unverified'] else (0,165,255)
   cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),color,5)
   cv2.putText(im,x['track_id'],(b[0],max(30,b[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2,cv2.LINE_AA)
  im=cv2.resize(im,(960,540));cv2.putText(im,f'frame {f}',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3,cv2.LINE_AA);cells.append(im)
 blank=255*cv2.UMat(540,960,cv2.CV_8UC3).get()
 while len(cells)%3:cells.append(blank)
 rows=[cv2.hconcat(cells[i:i+3]) for i in range(0,len(cells),3)]
 cv2.imwrite(str(a.output_dir/'object-presence-review-contact-sheet.png'),cv2.vconcat(rows))
 payload=json.dumps(d['annotations'])
 html='''<!doctype html><meta charset="utf-8"><title>Drone object-presence review</title><style>body{font:14px system-ui;margin:1rem}#stage{position:relative;width:min(96vw,1200px)}img{width:100%;display:block}.b{position:absolute;border:2px solid #f90;color:#f90;font-size:10px;pointer-events:none}.ok{border-color:#0b0;color:#0b0}select{margin:.5rem}</style><h1>Offline candidate overlay</h1><p>Orange: unverified visual/geometric candidate. Green: score-confirmed match. This is not organizer ground truth.</p><label>Frame <select id=f></select></label><div id=stage><img id=i></div><script>const A=__PAYLOAD__,F=[...new Set(A.map(x=>x.frame))].sort((a,b)=>a-b),s=document.querySelector('#f'),im=document.querySelector('#i'),st=document.querySelector('#stage');for(const x of F)s.add(new Option(x,x));function draw(){let f=+s.value;im.src=`../reconstructed-validation/frame_${String(f).padStart(6,'0')}.png`;im.onload=()=>{st.querySelectorAll('.b').forEach(x=>x.remove());for(const a of A.filter(x=>x.frame===f)){let b=a.bbox_normalized_xyxy,e=document.createElement('div');e.className='b '+(!a.unverified?'ok':'');e.style.left=(b[0]*100)+'%';e.style.top=(b[1]*100)+'%';e.style.width=((b[2]-b[0])*100)+'%';e.style.height=((b[3]-b[1])*100)+'%';e.textContent=a.track_id;st.append(e)}}}s.onchange=draw;s.value=140;draw()</script>'''.replace('__PAYLOAD__',payload)
 (a.output_dir/'object-presence-review.html').write_text(html)
if __name__=='__main__':main()
