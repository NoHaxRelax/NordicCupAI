"""Annotation-free confirmation across rectilinear crops of the SAME frame.

Inputs are model predictions and known crop geometry only. This assumes the
views share a source image coordinate system; it does not solve parallax or
register independent rendered perspectives. No model scores are boosted.
"""
import math

def overlap(a,b):
 area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
 return area/union if union>0 else 0

def cluster_views(views,overlap_threshold=.5):
 frames={};identities=set()
 for view in views:
  identity=(view['frame'],view['view'])
  if identity in identities:raise ValueError('Duplicate view identity')
  identities.add(identity);w,h=view['dimensions'];x,y,x2,y2=view['source_region']
  if w<=0 or h<=0 or x2<=x or y2<=y:raise ValueError('Invalid crop geometry')
  sx,sy=(x2-x)/w,(y2-y)/h
  if abs(sx-sy)>1e-8:raise ValueError('Nonuniform crop scaling')
  preds=frames.setdefault(view['frame'],[])
  for p in view['predictions']:
   a,b,c,d=p['bbox'];score=p['score']
   if not all(math.isfinite(v) for v in [a,b,c,d,score]) or not 0<=score<=1 or c<=a or d<=b:raise ValueError('Invalid prediction')
   preds.append(dict(p,bbox=[x+a*sx,y+b*sy,x+c*sx,y+d*sy],view=view['view'],zoom=view['zoom']))
 results=[]
 for frame,predictions in frames.items():
  remaining=sorted(predictions,key=lambda p:-p['score']);groups=[]
  while remaining:
   anchor=remaining.pop(0);group=[anchor];rest=[]
   for p in remaining:
    (group if p['class']==anchor['class'] and overlap(p['bbox'],anchor['bbox'])>=overlap_threshold else rest).append(p)
   remaining=rest;groups.append(dict(anchor,support=[dict(view=p['view'],zoom=p['zoom'],score=p['score']) for p in group]))
  results.append(dict(frame=frame,predictions=groups))
 return results

def confirm(clusters,threshold=.1,min_views=2,min_zooms=2):
 if not 0<=threshold<=1 or min_views<1 or min_zooms<1:raise ValueError('Invalid confirmation rule')
 output=[]
 for row in clusters:
  predictions=[]
  for p in row['predictions']:
   supporters=[s for s in p['support'] if s['score']>=threshold]
   if len({s['view'] for s in supporters})>=min_views and len({s['zoom'] for s in supporters})>=min_zooms:predictions.append(p)
  output.append(dict(frame=row['frame'],predictions=predictions))
 return output
