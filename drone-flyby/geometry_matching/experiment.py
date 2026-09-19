"""Offline training-only sparse 3D retrieval. Validation labels are read only by evaluate()."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, json, math, importlib.util
import cv2
import numpy as np
from scipy.optimize import least_squares

W,H=3840,2160

def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')

class Camera:
    def __init__(self, f=3140., pitch=18.25, yaw=0., step=13.8888889, altitude=600.):
        self.f,self.pitch,self.yaw,self.step,self.altitude=f,pitch,yaw,step,altitude
        p,y=np.radians([pitch,yaw]); c,s=np.cos(p),np.sin(p)
        base=np.array([[1,0,0],[0,-c,-s],[0,s,-c]])
        self.R=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])@base
        self.K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
    def centre(self,t): return np.array([0,t*self.step,self.altitude])
    def matrix(self,t): return self.K@np.c_[self.R,-self.R@self.centre(t)]
    def project(self,xyz,t):
        xyz=np.atleast_2d(xyz); p=(xyz-self.centre(t))@self.R.T@self.K.T
        return p[:,:2]/p[:,2:]
    def unproject(self,xy,t,z=0):
        xy=np.atleast_2d(xy); rays=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(self.K).T@self.R
        distance=(np.asarray(z)-self.altitude)/rays[:,2]
        return self.centre(t)+rays*distance[:,None]
    def json(self): return dict(f=self.f,pitch=self.pitch,yaw=self.yaw,step=self.step,altitude=self.altitude)

def triangulate(camera, xy, frames):
    A=[]
    for uv,t in zip(xy,frames):
        p=camera.matrix(t); A.extend([uv[0]*p[2]-p[0],uv[1]*p[2]-p[1]])
    _,_,v=np.linalg.svd(A); return v[-1,:3]/v[-1,3]

def rootsift(d):
    if d is None: return np.empty((0,128),np.float32)
    return np.sqrt(d/(d.sum(axis=1,keepdims=True)+1e-9)).astype(np.float32)

def sift_image(im,mask=None):
    sift=cv2.SIFT_create(nfeatures=16000,contrastThreshold=.018,edgeThreshold=14)
    kp,d=sift.detectAndCompute(im,mask)
    return np.array([k.pt for k in kp],np.float32).reshape(-1,2),rootsift(d)

def calibration(args,out):
    spec=importlib.util.spec_from_file_location('tilt',args.data/'elias/data/tilt_study.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=str(args.data)
    fit=mod.fit_camera(mod.load_tracks('helsinki'),13.8888889/600)
    cam=Camera(fit['f_px'],fit['pitch_deg'],fit['yaw_deg'])
    save(out/'camera_training.json',dict(parameters=cam.json(),fit=fit))
    return cam

def ply(path,xyz,rgb):
    with open(path,'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'%len(xyz))
        for p,c in zip(xyz,rgb): f.write('%.4f %.4f %.4f %d %d %d\n'%(*p,*c))

def build(args,out,cam):
    tracks=defaultdict(list); ann_by_frame={}
    for p in sorted((args.data/'src/helsinki/annotations').glob('*.json')):
        j=json.loads(p.read_text());ann_by_frame[j['frame']]=j['annotations']
        for a in j['annotations']:
            b=np.array(a['bbox']);
            if np.all(b[:2]>2) and b[2]<W-3 and b[3]<H-3: tracks[a['object_id']].append((j['frame'],b))
    bank=json.loads((args.data/'elias/sprites/bank.json').read_text())['sprites']
    masks={(s['class_name'],s['frame']):s for s in bank if s['source']=='helsinki'}
    features=defaultdict(list)
    for t in sorted(ann_by_frame):
        im=cv2.imread(str(args.data/f'src/helsinki/images/frame_{t:06d}.png'))
        for name,obs in tracks.items():
            found=[b for f,b in obs if f==t]
            if not found: continue
            box=found[0]; x1,y1=np.maximum(0,box[:2]-8);x2,y2=np.minimum([W,H],box[2:]+8)
            crop=im[y1:y2,x1:x2];mask=np.zeros(crop.shape[:2],np.uint8)
            s=masks.get((name,t))
            if s:
                spr=cv2.imread(str(args.data/'elias/sprites'/s['file']),cv2.IMREAD_UNCHANGED)
                sx,sy=s['origin_src']; a=spr[:,:,3]
                # Clip source mask into crop bounds.
                xa,ya=max(x1,sx),max(y1,sy);xb,yb=min(x2,sx+a.shape[1]),min(y2,sy+a.shape[0])
                if xb>xa and yb>ya:mask[ya-y1:yb-y1,xa-x1:xb-x1]=a[ya-sy:yb-sy,xa-sx:xb-sx]
            else:
                gc=np.zeros(crop.shape[:2],np.uint8)
                rect=(int(box[0]-x1),int(box[1]-y1),int(box[2]-box[0]),int(box[3]-box[1]))
                try:
                    cv2.grabCut(crop,gc,rect,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_RECT)
                    mask[np.isin(gc,[1,3])]=255
                except cv2.error: pass
            xy,d=sift_image(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),mask)
            colors=crop[np.clip(xy[:,1].astype(int),0,len(crop)-1),np.clip(xy[:,0].astype(int),0,crop.shape[1]-1),::-1]
            features[name].append(dict(t=t,xy=xy+[x1,y1],d=d,colors=colors,box=box))
    all_xyz=[];all_d=[];all_ids=[];meta=[];models=[]
    matcher=cv2.BFMatcher()
    for name,rows in sorted(features.items()):
        nodes=[];dlist=[];colors=[]; parents=[]; offsets=[]
        for row in rows:
            offsets.append(len(nodes))
            nodes.extend([(row['t'],p) for p in row['xy']]);dlist.extend(row['d']);colors.extend(row['colors'])
        parents=list(range(len(nodes)))
        def root(x):
            while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
            return x
        for i,a in enumerate(rows):
            for j in range(i+1,len(rows)):
                b=rows[j]
                if not 1<=b['t']-a['t']<=6 or len(a['d'])<2 or len(b['d'])<2:continue
                matches=matcher.knnMatch(a['d'],b['d'],k=2)
                for m,n in matches:
                    if m.distance>.72*n.distance:continue
                    pa,pb=a['xy'][m.queryIdx],b['xy'][m.trainIdx]
                    xyz=triangulate(cam,[pa,pb],[a['t'],b['t']])
                    if not -15<xyz[2]<120:continue
                    err=np.linalg.norm(cam.project(xyz,a['t'])[0]-pa)+np.linalg.norm(cam.project(xyz,b['t'])[0]-pb)
                    if err>2.5:continue
                    aa,bb=root(offsets[i]+m.queryIdx),root(offsets[j]+m.trainIdx)
                    parents[aa]=bb
        components=defaultdict(list)
        for i in range(len(nodes)): components[root(i)].append(i)
        points=[]; pointcols=[];entries=[]; reproj=[]
        for ids in components.values():
            frames=np.array([nodes[i][0] for i in ids]);xy=np.array([nodes[i][1] for i in ids])
            if len(set(frames))<3 or np.ptp(frames)<3 or len(set(frames))!=len(frames):continue
            xyz=triangulate(cam,xy,frames)
            err=np.array([np.linalg.norm(cam.project(xyz,t)[0]-p) for t,p in zip(frames,xy)])
            if not -15<xyz[2]<120 or np.max(err)>3 or np.sqrt(np.mean(err**2))>1.5:continue
            pointid=len(points);points.append(xyz);pointcols.append(np.median(np.array(colors)[ids],axis=0));reproj.extend(err)
            entries.extend((i,pointid) for i in ids)
        if not points:
            models.append(dict(name=name,landmarks=0,observations=len(nodes)));continue
        points=np.array(points); centre=np.r_[np.median(points[:,:2],axis=0),0.]
        local=points-centre; global_start=len(all_xyz);all_xyz.extend(local)
        for i,pid in entries:all_d.append(dlist[i]);all_ids.append(global_start+pid)
        # Observed boxes become per-view training support; retain them for projected template boxes.
        modelid=len(models)
        boxrows=[]
        for row in rows:
            boxrows.append(dict(frame=row['t'],box=row['box'].tolist()))
        models.append(dict(name=name,landmarks=len(points),observations=len(nodes),centre=centre.tolist(),point_start=global_start,point_end=len(all_xyz),boxes=boxrows,reprojection_median_px=float(np.median(reproj)),height_range=[float(points[:,2].min()),float(points[:,2].max())]))
        meta.extend([name]*len(points));ply(out/f'{name}.ply',local,np.array(pointcols,dtype=int))
        print('model',name,len(points),'landmarks',flush=True)
    save(out/'models.json',models)
    np.savez_compressed(out/'bank.npz',xyz=np.array(all_xyz),descriptors=np.array(all_d,np.float32),landmark_ids=np.array(all_ids),classes=np.array(meta))
    return models

def validation_calibration(args,out,traincam):
    pairs=[]
    for t in [8,35,70,110,150,190,230]:
        ims=[cv2.imread(str(args.data/f'validation/reconstructed-validation/frame_{f:06d}.png'),0) for f in [t,t+3]]
        features=[sift_image(cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA)) for im in ims]
        (pa,da),(pb,db)=features
        good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.65*n.distance]
        xa=np.array([pa[m.queryIdx]*2 for m in good]);xb=np.array([pb[m.trainIdx]*2 for m in good])
        hom,mask=cv2.findHomography(xa,xb,cv2.RANSAC,3.)
        keep=mask[:,0]>0;xa,xb=xa[keep][::5],xb[keep][::5]
        pairs.append((t,xa,xb))
    def residual(p,split):
        c=Camera(p[0],p[1],p[2],traincam.step);res=[]
        for t,a,b in pairs:
            a,b=a[split::2],b[split::2]
            res.extend((c.project(c.unproject(a,t),t+3)-b).ravel())
        return np.array(res)
    sol=least_squares(lambda p:residual(p,0),[3340,20,0],bounds=([2700,12,-5],[3700,26,5]),loss='soft_l1',f_scale=2.)
    cam=Camera(*sol.x)
    test=residual(sol.x,1).reshape(-1,2)
    save(out/'camera_validation.json',dict(parameters=cam.json(),method='Known step 13.8888889m and assumed altitude 600m; fit focal length, pitch and yaw to background SIFT three-frame displacements, alternating held-out correspondences. No validation labels. Effective height and focal length are ambiguous on non-flat terrain.',heldout_n=len(test),heldout_median_px=float(np.median(np.linalg.norm(test,axis=1))),heldout_p90_px=float(np.percentile(np.linalg.norm(test,axis=1),90)),nominal_camera_median_px=float(np.median(np.linalg.norm(residual([traincam.f,traincam.pitch,traincam.yaw],1).reshape(-1,2),axis=1)))))
    return cam

def scan(args,out,cam):
    bank=np.load(out/'bank.npz'); desc=bank['descriptors'];pids=bank['landmark_ids'];xyz=bank['xyz'];classes=bank['classes']
    flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=80));flann.add([desc]);flann.train()
    fp=open(out/'correspondences.jsonl','w');counts=[]
    cache=out/'features';cache.mkdir(exist_ok=True)
    for t in range(1,250):
        path=cache/f'{t:03d}.npz'
        if path.exists():
            z=np.load(path);xy,d=z['xy'],z['d']
        else:
            im=cv2.imread(str(args.data/f'validation/reconstructed-validation/frame_{t:06d}.png'),0)
            xy,d=sift_image(im);np.savez_compressed(path,xy=xy,d=d)
        if len(d)==0:
            counts.append(dict(frame=t,features=0,matches=0));continue
        matches=flann.knnMatch(d,k=8);rows=[]
        for q,mm in enumerate(matches):
            m=mm[0];pid=int(pids[m.trainIdx]);other=next((n for n in mm[1:] if pids[n.trainIdx]!=pid and (classes[pids[n.trainIdx]]!=classes[pid] or np.linalg.norm(xyz[pids[n.trainIdx]]-xyz[pid])>1.)),None)
            if other is None or m.distance>.78*other.distance or m.distance>.55:continue
            row=dict(frame=t,xy=xy[q].tolist(),landmark=pid,class_name=str(classes[pid]),distance=float(m.distance),ratio=float(m.distance/other.distance))
            fp.write(json.dumps(row)+'\n');rows.append(row)
        fp.flush();counts.append(dict(frame=t,features=len(d),matches=len(rows)))
        if t%10==0:print('scan',t,'features',len(d),'matches',len(rows),flush=True)
    fp.close();save(out/'scan_counts.json',counts)

def rigid(a,b):
    ac=a-a.mean(0);bc=b-b.mean(0);u,_,vt=np.linalg.svd(ac.T@bc);r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    return r,b.mean(0)-a.mean(0)@r.T

def fit_scene(args,out,cam,traincam):
    bank=np.load(out/'bank.npz');xyz=bank['xyz'];models=json.loads((out/'models.json').read_text());byname={m['name']:m for m in models}
    groups=defaultdict(list)
    for line in (out/'correspondences.jsonl').read_text().splitlines():
        row=json.loads(line);groups[row['class_name']].append(row)
    rng=np.random.default_rng(19);detections=[]
    for name,rows in groups.items():
        if len(rows)<6:continue
        ids=np.array([r['landmark'] for r in rows]);a=xyz[ids,:2]
        # Each correspondence intersects the ray at its training landmark height.
        b=np.array([cam.unproject([r['xy']],r['frame'],xyz[i,2])[0,:2] for r,i in zip(rows,ids)])
        ts=np.array([r['frame'] for r in rows]);available=np.ones(len(rows),bool)
        for instance in range(40):
            active=np.where(available)[0]
            if len(active)<6:break
            best=[]
            for _ in range(min(20000,max(2000,len(active)*20))):
                i,j=rng.choice(active,2,replace=False)
                da=a[j]-a[i];db=b[j]-b[i]
                na,nb=np.linalg.norm(da),np.linalg.norm(db)
                if na<.6 or abs(na-nb)>max(.6,na*.08) or abs(ts[i]-ts[j])>40:continue
                theta=np.arctan2(db[1],db[0])-np.arctan2(da[1],da[0]);c,s=np.cos(theta),np.sin(theta);r=np.array([[c,-s],[s,c]]);tr=b[i]-a[i]@r.T
                err=np.linalg.norm(a@r.T+tr-b,axis=1);inliers=np.where((err<.8)&available)[0]
                if len(inliers)>len(best):best=inliers
            if len(best)<6:break
            r,tr=rigid(a[best],b[best]);err=np.linalg.norm(a@r.T+tr-b,axis=1);best=np.where((err<1.)&available)[0]
            available[best]=False
            if len(set(ts[best]))<3 or len(set(ids[best]))<6 or np.ptp(ts[best])<3:continue
            r,tr=rigid(a[best],b[best]);error=np.linalg.norm(a[best]@r.T+tr-b[best],axis=1)
            model=byname[name];points=xyz[model['point_start']:model['point_end']].copy();points[:,:2]=points[:,:2]@r.T+tr
            # The point cloud is sparse: preserve a separately estimated 3D bounding prism from training boxes.
            centre=np.array(model['centre']); lo=[];hi=[]
            height=float(np.clip(np.percentile(points[:,2],90),0,70))
            for br in model['boxes']:
                bb=br['box'];uv=np.array([[bb[0],bb[1]],[bb[2],bb[1]],[bb[2],bb[3]],[bb[0],bb[3]]])
                ground=traincam.unproject(uv,br['frame'],max(0,height/2))-centre
                lo.append(ground[:,:2].min(0));hi.append(ground[:,:2].max(0))
            low=np.percentile(lo,35,axis=0);high=np.percentile(hi,65,axis=0)
            prism=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [0,height]])
            prism[:,:2]=prism[:,:2]@r.T+tr
            frames=[]
            for t in range(max(1,int(ts[best].min())-5),min(249,int(ts[best].max())+5)+1):
                p=cam.project(prism,t);bb=np.r_[p.min(0),p.max(0)];bb=np.clip(bb,[0,0,0,0],[W,H,W,H])
                if np.all(bb[2:]-bb[:2]>3):frames.append(dict(frame=t,bbox=bb.tolist()))
            detection=dict(id=f'{name}-{instance:02d}',class_name=name,position_m=tr.tolist(),yaw_deg=float(np.degrees(np.arctan2(r[1,0],r[0,0]))),inlier_correspondences=len(best),distinct_landmarks=len(set(ids[best])),distinct_frames=len(set(ts[best])),first_frame=int(ts[best].min()),last_frame=int(ts[best].max()),median_error_m=float(np.median(error)),score=float(min(.99,len(set(ids[best]))/10)*min(1,len(set(ts[best]))/5)),frames=frames,evidence=[rows[i] for i in best])
            detections.append(detection);print('instance',detection['id'],detection['distinct_landmarks'],detection['distinct_frames'],flush=True)
    save(out/'detections.json',detections)
    save(out/'scene.json',dict(camera=cam.json(),coordinate_system='metres: X right, Y along flight, Z up; camera C(t)=(0,step*t,600)',instances=[{k:v for k,v in d.items() if k not in ['frames','evidence']} for d in detections],warning='Sparse observed surfaces and approximate bounding prisms; not watertight meshes.'))

def iou(a,b):
    a,b=np.array(a),np.array(b);inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter+1e-9))

def evaluate(args,out):
    detections=json.loads((out/'detections.json').read_text()); truth=[]
    for p in sorted((args.data/'src/validation/annotations').glob('*.json')):
        j=json.loads(p.read_text())
        for a,prov in zip(j['annotations'],j['provenance']):
            if prov['review_status']=='directly_reviewed_track':truth.append(dict(frame=j['frame'],name=a['object_id'],box=a['bbox'],track=prov['track_id']))
    predictions=[dict(frame=f['frame'],name=d['class_name'],box=f['bbox'],score=d['score'],instance=d['id']) for d in detections for f in d['frames']]
    used=set();tp=0;fp=0
    for p in sorted(predictions,key=lambda p:-p['score']):
        candidates=[(iou(p['box'],g['box']),i) for i,g in enumerate(truth) if i not in used and g['name']==p['name'] and g['frame']==p['frame']]
        best=max(candidates,default=(0,-1))
        if best[0]>=.5:tp+=1;used.add(best[1])
        else:fp+=1
    retrieval=[]
    for d in detections:
        matches=Counter()
        for e in d['evidence']:
            for g in truth:
                if e['frame']==g['frame'] and g['name']==d['class_name'] and g['box'][0]<=e['xy'][0]<=g['box'][2] and g['box'][1]<=e['xy'][1]<=g['box'][3]:matches[g['track']]+=1
        retrieval.append(dict(id=d['id'],overlapping_reviewed_tracks=dict(matches)))
    save(out/'evaluation.json',dict(reference='Participant directly-reviewed validation tracks only; incomplete and possibly mislabelled, not organizer ground truth. Unmatched detections are provisional false positives against this reference.',reviewed_boxes=len(truth),reviewed_tracks=len(set(g['track'] for g in truth)),detected_instances=len(detections),predicted_boxes=len(predictions),tp50=tp,unmatched_predictions=fp,fn50=len(truth)-tp,precision50=tp/max(1,tp+fp),recall50=tp/max(1,len(truth)),retrieval=retrieval))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['build','scan','fit','evaluate','all'],default='all');args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4);cv2.setRNGSeed(19)
    if args.stage in ['build','all']:
        cam=calibration(args,out);build(args,out,cam);validation_calibration(args,out,cam)
    traincam=Camera(**json.loads((out/'camera_training.json').read_text())['parameters']);cam=Camera(**json.loads((out/'camera_validation.json').read_text())['parameters'])
    if args.stage in ['scan','all']:scan(args,out,cam)
    if args.stage in ['fit','all']:fit_scene(args,out,cam,traincam)
    if args.stage in ['evaluate','all']:evaluate(args,out)
