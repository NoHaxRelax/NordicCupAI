"""Training-only rotation/brightness tolerant proposals plus temporal verification.
This is a 2D/ground-plane fallback, not reconstructed volumetric geometry.
"""
from pathlib import Path
import json,argparse,math,time
from collections import defaultdict
import cv2,numpy as np
from experiment import Camera,save,iou

def rotate(image,mask,angle,scale):
    h,w=image.shape;c=np.cos(np.radians(angle));s=np.sin(np.radians(angle));ow=int(np.ceil(scale*(abs(c)*w+abs(s)*h)))+4;oh=int(np.ceil(scale*(abs(s)*w+abs(c)*h)))+4
    m=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,scale);m[:,2]+=[(ow-w)/2,(oh-h)/2]
    return cv2.warpAffine(image,m,(ow,oh)),cv2.warpAffine(mask,m,(ow,oh),flags=cv2.INTER_NEAREST)

def feature(gray):
    gray=gray.astype(np.float32);return gray-cv2.GaussianBlur(gray,(0,0),3.)

def main(args):
    cv2.setNumThreads(2);out=args.out;cam=Camera(**json.loads((out/'camera_validation.json').read_text())['parameters'])
    models=json.loads((out/'models.json').read_text());names=[m['name'] for m in models if (m['landmarks']>=3 if args.complement else m['landmarks']<3)]
    bank=json.loads((args.data/'elias/sprites/bank.json').read_text())['sprites'];templates=[];sources=[]
    for name in names:
        options=[s for s in bank if s['source']=='helsinki' and s['class_name']==name]
        if options:
            s=min(options,key=lambda s:abs(s['center_src'][1]-650));spr=cv2.imread(str(args.data/'elias/sprites'/s['file']),-1);gray=cv2.cvtColor(spr[:,:,:3],cv2.COLOR_BGR2GRAY);mask=spr[:,:,3];source=dict(name=name,frame=s['frame'],file=s['file'],mask='training sprite')
        else:
            opts=[]
            for p in (args.data/'src/helsinki/annotations').glob('*.json'):
                j=json.loads(p.read_text())
                for a in j['annotations']:
                    x1,y1,x2,y2=a['bbox']
                    if a['object_id']==name and x1>2 and y1>2 and x2<3837 and y2<2157:opts.append((abs((y1+y2)/2-650),j['frame'],a['bbox']))
            if not opts:continue
            _,f,b=min(opts);im=cv2.imread(str(args.data/f'src/helsinki/images/frame_{f:06d}.png'))
            x1,y1,x2,y2=b;crop=im[y1-5:y2+5,x1-5:x2+5];gc=np.zeros(crop.shape[:2],np.uint8)
            cv2.grabCut(crop,gc,(5,5,x2-x1,y2-y1),np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_RECT)
            mask=np.uint8(np.isin(gc,[1,3]))*255;gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);source=dict(name=name,frame=f,box=b,mask='GrabCut from organizer training box')
            if np.sum(mask>0)<15:mask[5:-5,5:-5]=255;source['mask']='organizer training box rectangle (fallback)'
        # Use foreground texture and silhouette: fill exterior with foreground mean before high pass.
        gray=np.where(mask>0,gray,np.mean(gray[mask>0])).astype(np.float32);base=feature(gray);base*=mask/255.
        for angle in range(0,360,15):
            for scale in [.85,1.,1.15]:
                full,fullmask=rotate(base,mask,angle,scale)
                small=cv2.resize(full,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
                if min(small.shape)<5 or np.std(small)<1:continue
                templates.append(dict(name=name,angle=angle,scale=scale,small=small,full=full,mask=fullmask))
        sources.append(source)
    save(out/(args.prefix+'_sources.json'),sources)
    proposals=json.loads((out/(args.prefix+"_proposals.json")).read_text()) if args.resume and (out/(args.prefix+"_proposals.json")).exists() else [];times=[]
    done=max([p["frame"] for p in proposals],default=0)
    for frame in [5]+list(range(9,250,4)):
        if frame<=done:continue
        im=cv2.imread(str(args.data/f'validation/reconstructed-validation/frame_{frame:06d}.png'),0)
        if frame!=5: im=im[:500]
        search=feature(cv2.resize(im,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA));start=time.monotonic();cand=[]
        for ti,temp in enumerate(templates):
            t=temp['small'];h,w=t.shape
            if h>=search.shape[0] or w>=search.shape[1]:continue
            scores=cv2.matchTemplate(search,t,cv2.TM_CCOEFF_NORMED)
            # Only strongest response per rotated template per sampled frame.
            _,score,_,loc=cv2.minMaxLoc(scores)
            if score<.47:continue
            x,y=loc;box=[x*2,y*2,(x+w)*2,(y+h)*2]
            cand.append(dict(frame=frame,name=temp['name'],template=ti,box=box,score=float(score)))
        cand.sort(key=lambda x:-x['score']);kept=[]
        for p in cand:
            if any(p['name']==q['name'] and iou(p['box'],q['box'])>.3 for q in kept):continue
            kept.append(p)
        proposals.extend(kept);times.append(time.monotonic()-start)
        save(out/(args.prefix+'_proposals.json'),proposals)
        print('template scan',frame,'proposals',len(kept),'seconds',round(times[-1],2),flush=True)
    # Verify fixed orientation across five frames with a small local alignment search.
    verified=[];imagecache={}
    for pi,p in enumerate(sorted(proposals,key=lambda p:-p['score'])):
        box=np.array(p['box']);xy=(box[:2]+box[2:])/2;world=cam.unproject(xy,p['frame'])[0]
        if any(p['name']==q['name'] and np.linalg.norm(world[:2]-np.array(q['position_m']))<5 for q in verified):continue
        temp=templates[p['template']];scores=[];frames=[]
        for frame in range(max(1,p['frame']-4),min(249,p['frame']+4)+1,2):
            centre=cam.project(world,frame)[0];t=temp['full'];h,w=t.shape
            x,y=np.round(centre-[w/2,h/2]).astype(int);pad=10
            if x-pad<0 or y-pad<0 or x+w+pad>=3840 or y+h+pad>=2160:continue
            if frame not in imagecache:
                imagecache[frame]=cv2.imread(str(args.data/f'validation/reconstructed-validation/frame_{frame:06d}.png'),0)

            crop=feature(imagecache[frame][y-pad:y+h+pad,x-pad:x+w+pad]);response=cv2.matchTemplate(crop,t,cv2.TM_CCOEFF_NORMED)
            _,score,_,(dx,dy)=cv2.minMaxLoc(response);scores.append(float(score));frames.append(dict(frame=frame,bbox=[int(x-pad+dx),int(y-pad+dy),int(x-pad+dx+w),int(y-pad+dy+h)],score=float(score)))
        if len(scores)>=3 and np.median(scores)>=.40 and sum(s>=.38 for s in scores)>=3:
            verified.append(dict(id=f"fallback-{len(verified):03d}",name=p['name'],position_m=world[:2].tolist(),angle=temp['angle'],scale=temp['scale'],proposal=p,median_ncc=float(np.median(scores)),frames=frames))
        if pi%100==0:print('verify',pi,'accepted',len(verified),flush=True)
    save(out/(args.prefix+'_verified.json'),verified)
    save(out/(args.prefix+'_summary.json'),dict(classes=names,templates=len(templates),proposal_count=len(proposals),verified_count=len(verified),native_input=True,proposal_search_scale=.5,verification_scale=1.,sampling='Full frame 5 then top 500 rows every four frames; temporal verification uses native-resolution crops. Discovery assumes static objects entering the top of the frame.',thresholds=dict(proposal_ncc=.47,median_native_ncc=.4,minimum_supported_frames=3),warning='Exploratory thresholds, not calibrated probabilities. 2D rotated templates do not model newly exposed surfaces of tall objects.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--complement',action='store_true');p.add_argument('--prefix',default='fallback');p.add_argument('--resume',action='store_true');main(p.parse_args())
