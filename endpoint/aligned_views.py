"""Align overlapping delivered views without allocating full coordinate grids."""
import cv2
import numpy as np

def patches(first,second,first_view,second_view,min_size=48):
 a,b=np.array(first_view.region),np.array(second_view.region)
 region=np.r_[np.maximum(a[:2],b[:2]),np.minimum(a[2:],b[2:])]
 scale=np.maximum(first_view.scale,second_view.scale)
 size=np.floor((region[2:]-region[:2])/scale).astype(int)
 if np.any(size<min_size):return None,region,scale,size
 result=[]
 for image,view in ((first,first_view),(second,second_view)):
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
  lo=(region[:2]-view.region[:2])/view.scale
  extent=size*scale/view.scale
  if np.allclose(lo,np.round(lo),atol=1e-6) and np.allclose(extent,np.round(extent),atol=1e-6):
   x,y=np.round(lo).astype(int);w,h=np.round(extent).astype(int)
   patch=gray[y:y+h,x:x+w]
   if (w,h)!=tuple(size):patch=cv2.resize(patch,tuple(size),interpolation=cv2.INTER_LINEAR)
   result.append(patch)
  else:
   yy,xx=np.indices(size[::-1],dtype=np.float32)
   local_x=(xx+.5)*scale[0]/view.scale[0]+lo[0]-.5
   local_y=(yy+.5)*scale[1]/view.scale[1]+lo[1]-.5
   result.append(cv2.remap(gray,local_x.astype(np.float32),local_y.astype(np.float32),cv2.INTER_LINEAR))
 return result,region,scale,size
