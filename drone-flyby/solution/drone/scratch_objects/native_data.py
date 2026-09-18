"""Add real reference tiles to counter synthetic cutout boundaries.

No additional validation pixels. Only reference frames and positive crops already
represented by the bank qualify. Tiles containing any excluded/partial source
object are rejected, rather than treating that object as background.
"""
import argparse
import json
from pathlib import Path
import shutil
import cv2
from .prepare import sha,overlap
from .evaluate import starts


def build(root,base,output):
    manifest=json.loads((base/'manifest.json').read_text())
    bank=manifest['bank']
    references={(r['frame'],r['class']) for r in bank['templates'] if r['source']=='organizer_reference'}
    output.mkdir(parents=True,exist_ok=False)
    for name in ['images','labels']:shutil.copytree(base/name,output/name)
    records=manifest['records'].copy()
    count=0
    for frame in sorted({f for f,c in references}):
        source=root/'data/drone/reference/helsinki'
        path=source/'images'/f'frame_{frame:06d}.png'
        annotation=source/'annotations'/f'frame_{frame:06d}.json'
        for p in [path,annotation]:
            if sha(p)!=bank['source_hashes'][str(p.relative_to(root))]:raise ValueError('Reference source changed')
        image=cv2.imread(str(path))
        annotations=json.loads(annotation.read_text())['annotations']
        for y in starts(2160,540):
            for x in starts(3840,960):
                region=[x,y,x+960,y+540]
                labels=[];objects=[];reject=False
                for a in annotations:
                    b=a['bbox'];inter=overlap(region,b)
                    if not inter:continue
                    area=(b[2]-b[0])*(b[3]-b[1])
                    if (frame,a['object_id']) not in references or inter/area<.65:
                        reject=True;break
                    bx1=max(x,b[0])-x;by1=max(y,b[1])-y;bx2=min(x+960,b[2])-x;by2=min(y+540,b[3])-y
                    label=manifest['classes'].index(a['object_id'])
                    labels.append(f'{label} {(bx1+bx2)/1920:.7f} {(by1+by2)/1080:.7f} {(bx2-bx1)/960:.7f} {(by2-by1)/540:.7f}')
                    objects.append(dict(template=f"{a['object_id']}-{frame:06d}",bbox=[bx1,by1,bx2,by2]))
                if reject:continue
                name=f'real-{frame:06d}-{x}-{y}'
                dest=output/'images/train'/f'{name}.jpg'
                label_path=output/'labels/train'/f'{name}.txt'
                cv2.imwrite(str(dest),image[y:y+540,x:x+960],[cv2.IMWRITE_JPEG_QUALITY,95])
                label_path.write_text('\n'.join(labels)+'\n' if labels else '')
                records.append(dict(split='train',file=str(dest.relative_to(output)),sha256=sha(dest),label=str(label_path.relative_to(output)),
                                    label_sha256=sha(label_path),reference_frame=frame,region=region,objects=objects))
                count+=1
    manifest.update(records=records,base_manifest_sha256=sha(base/'manifest.json'),native_reference_tiles=count,
                    policy=manifest['policy']+' Plus native reference tiles containing only bank-represented positives, with context; no new validation pixels.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(native_reference_tiles=count,total_training=sum(r['split']=='train' for r in records))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.root,a.base,a.output)
