"""Compare confidence rankings at equal hit counts on the same development views."""
import argparse
from pathlib import Path
from common import read,write
from evaluate_views import iou

def frontier(rows):
 events=[];targets=sum(len(r['truth']) for r in rows)
 for row in rows:
  used=set()
  for p in sorted(row['predictions'],key=lambda p:-p['score']):
   candidates=[(iou(p['bbox'],t['bbox']),i) for i,t in enumerate(row['truth']) if i not in used and p['class']==t['class']]
   overlap,index=max(candidates,default=(0,-1));hit=overlap>=.5
   if hit:used.add(index)
   events.append((p['score'],int(hit)))
 groups={}
 for score,hit in events:
  v=groups.setdefault(score,[0,0]);v[0]+=hit;v[1]+=1-hit
 tp=fp=0;points={}
 for score,(hits,misses) in sorted(groups.items(),reverse=True):
  tp+=hits;fp+=misses
  if hits and tp not in points:points[tp]=dict(tp=tp,fp=fp,threshold=score)
 return dict(targets=targets,points=list(points.values()))

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();report={}
 for name in ['baseline','same_class_product','background_filter','relabel']:
  rows=read(a.directory/(name+'.json'))['results'];report[name]={'all':frontier(rows),**{f'L{z}':frontier([r for r in rows if r['zoom']==z]) for z in range(3)}}
 write(a.directory/'frontiers.json',report)
 print({n:r['all'] for n,r in report.items()})
if __name__=='__main__':main()
