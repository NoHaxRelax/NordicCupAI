"""Build box-transport checks and figures from analyze_scene_geometry.py output."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from analyze_scene_geometry import OUT, REF, VAL, iou, pair_study


def composed(lookup, start, end):
    h = np.eye(3)
    for k in range(min(start,end), max(start,end)):
        h = np.array(lookup['reference', k, k+1]['matrices']['homography']) @ h
    return h if start <= end else np.linalg.inv(h)


def transport(j, lookup):
    results = {}
    for gap in [1,3,5,10]:
        rows = []
        for name, track in j['reference_annotations']['unclipped_tracks'].items():
            boxes = dict(zip(track['frames'], track['boxes']))
            for f, box in boxes.items():
                if f+gap not in boxes:
                    continue
                # Use exactly the anchors in the three-observation velocity
                # baseline, so the method comparison has identical examples.
                if f-1 not in boxes or f-2 not in boxes:
                    continue
                x1,y1,x2,y2 = box
                corners = np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]])
                poly = cv2.perspectiveTransform(corners, composed(lookup,f,f+gap))[0]
                pred = np.r_[poly.min(0),poly.max(0)]
                true = np.array(boxes[f+gap])
                rows.append({'class':name,'frame':f,'iou':iou(pred,true),
                    'center_error_px':float(np.linalg.norm((pred[:2]+pred[2:]-true[:2]-true[2:])/2))})
        results[gap] = {'n':len(rows), 'median_iou':float(np.median([r['iou'] for r in rows])),
            'fraction_iou_ge_0_5':float(np.mean([r['iou']>=.5 for r in rows])),
            'center_error_median_px':float(np.median([r['center_error_px'] for r in rows])),
            'center_error_p90_px':float(np.percentile([r['center_error_px'] for r in rows],90)), 'rows':rows}
    (OUT/'box-transport.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def figures(j, lookup):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    row = lookup['validation',140,141]
    x=np.array(row['flow_probe_xy']);v=np.array(row['homography_flow_px'])
    im=Image.open(VAL/'frame_000140.png').convert('RGB')
    fig,ax=plt.subplots(figsize=(12,7))
    ax.imshow(im)
    ax.quiver(x[:,0],x[:,1],v[:,0],v[:,1],angles='xy',scale_units='xy',scale=1/3,
              color='#ffdf47',width=.0035)
    for p,vel in zip(x,v):
        ax.text(p[0],p[1]-60,f'{vel[0]:+.1f}, {vel[1]:+.1f} px',ha='center',fontsize=9,
                color='white',bbox={'facecolor':'#17232b','alpha':.85,'pad':4,'edgecolor':'none'})
    ax.set(xlim=(0,3840),ylim=(2160,0),xlabel='Source x (pixels)',ylabel='Source y (pixels)',
           title='Measured perspective flow: validation frame 140 to 141\nLabels are actual displacements; arrows enlarged 3 times')
    fig.tight_layout();fig.savefig(OUT/'motion-field.png',dpi=150);plt.close(fig)

    # Each class keeps a fixed crop size and display scale across time. Clipped
    # boxes excluded. These are observed pixels, not synthesized viewpoints.
    names=['ta-ta','small_tower','large_tower','helicopter']
    sheet=Image.new('RGB',(1100,4*220),'#17232b');d=ImageDraw.Draw(sheet)
    for row,name in enumerate(names):
        track=j['reference_annotations']['unclipped_tracks'][name]
        fs=track['frames'];bs=track['boxes'];side=round(max(max(b[2]-b[0],b[3]-b[1]) for b in bs)+20)
        d.text((8,row*220+8),name,fill='white')
        for col,i in enumerate([0,len(fs)//2,len(fs)-1]):
            f=fs[i];b=bs[i];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            with Image.open(REF/'images'/f'frame_{f:06d}.png') as im:
                p=im.crop((round(cx-side/2),round(cy-side/2),round(cx+side/2),round(cy+side/2))).resize((180,180))
            x=190+col*300;y=row*220;sheet.paste(p,(x,y));d.text((x,y+184),f'frame {f} | box {int(b[2]-b[0])} x {int(b[3]-b[1])} px',fill='white')
    sheet.save(OUT/'viewpoint-evidence.jpg')

    # Manually selected examples for appearance comparison, not new labels.
    examples=[('small_plane',0,127,(2725,1060,2815,1140)),
        ('jet_plane',12,66,(1050,800,1150,875)),('helicopter',0,45,(2490,110,2640,220)),
        ('small_tower',0,66,(2210,460,2270,550)),('large_launcher',12,140,(697,25,830,140))]
    sheet=Image.new('RGB',(760,len(examples)*220),'#17232b');d=ImageDraw.Draw(sheet)
    for row,(name,rf,vf,vb) in enumerate(examples):
        anns=json.loads((REF/'annotations'/f'frame_{rf:06d}.json').read_text())['annotations']
        rb=next(a['bbox'] for a in anns if a['object_id']==name)
        d.text((5,row*220+10),name,fill='white')
        for col,(folder,frame,box,tag) in enumerate([(REF/'images',rf,rb,'reference label'),(VAL,vf,vb,'validation appearance')]):
            x1,y1,x2,y2=box;side=max(x2-x1,y2-y1)+10;cx=(x1+x2)/2;cy=(y1+y2)/2
            with Image.open(folder/f'frame_{frame:06d}.png') as im:
                p=im.crop((round(cx-side/2),round(cy-side/2),round(cx+side/2),round(cy+side/2))).convert('RGB').resize((180,180))
            x=180+col*285;y=row*220;sheet.paste(p,(x,y));d.text((x,y+184),f'{tag} | frame {frame}',fill='white')
    sheet.save(OUT/'cross-sequence-appearance.jpg')

    # Register three observations to one ground-plane coordinate system. Black
    # areas are outside the earlier frame; they are not missing scene objects.
    h1=np.array(lookup['validation',130,140]['matrices']['homography'])
    h2=np.array(lookup['validation',140,150]['matrices']['homography'])
    regions=[('Curved building',(1400,650,2100,1350)),
             ('Tall building',(2040,740,2590,1290)),('Roofs',(1900,40,2600,740))]
    sheet=Image.new('RGB',(1500,1620),'#17232b');d=ImageDraw.Draw(sheet)
    aligned={}
    for f in [130,140,150]:
        im=cv2.imread(str(VAL/f'frame_{f:06d}.png'))
        if f!=140:
            im=cv2.warpPerspective(im,h1 if f==130 else np.linalg.inv(h2),(3840,2160))
        aligned[f]=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    for row,(label,box) in enumerate(regions):
        for col,f in enumerate([130,140,150]):
            sheet.paste(aligned[f].crop(box).resize((500,500)),(col*500,row*540))
            d.text((col*500+10,row*540+507),f'{label} | frame {f}, aligned to frame 140',fill='white')
    sheet.save(OUT/'validation-buildings-aligned.jpg')


def main():
    j=json.loads((OUT/'measurements.json').read_text())
    lookup={(r['sequence'],r['from'],r['to']):r for r in j['pairs']}
    transport(j,lookup);figures(j,lookup)
    # Independent whole-sequence check for near repeated images. Mean absolute
    # differences use thumbnails only; candidate freezes need geometric checks.
    rows=[];previous=None
    for f in range(5,250):
        with Image.open(VAL/f'frame_{f:06d}.png') as im:
            thumb=np.array(im.convert('RGB').resize((384,216)),dtype=np.float32)
        if previous is not None:
            rows.append({'from':f-1,'to':f,'thumbnail_mean_absolute_difference':float(np.abs(thumb-previous).mean())})
        previous=thumb
    (OUT/'temporal-differences.json').write_text(json.dumps(rows,indent=2)+'\n')
    candidates=[r for r in rows if r['thumbnail_mean_absolute_difference']<5]
    print('Near repeats (thumbnail RGB MAD < 5):', candidates)
    confirmations=[]
    for candidate in candidates:
        for a in [candidate['from'], candidate['to']]:
            if a >= 249:
                continue
            r=lookup.get(('validation',a,a+1))
            if r is None:
                r=pair_study(VAL/f'frame_{a:06d}.png',VAL/f'frame_{a+1:06d}.png','validation',a,a+1)
            confirmations.append(r)
    (OUT/'timing-confirmation.json').write_text(json.dumps(confirmations,indent=2)+'\n')


if __name__=='__main__':
    main()
