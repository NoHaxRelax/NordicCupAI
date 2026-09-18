"""Blind end-to-end hybrid search, including per-view verification and fusion."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from .hybrid import HybridDetector,HybridSettings
from .evaluate import starts,measure
from .prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--verifier',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cnn-threshold',type=float,default=.25)
    p.add_argument('--feature-upsample',type=float,default=2.)
    p.add_argument('--feature-ransac-iterations',type=int,default=3000)
    a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();detector=HybridDetector(a.weights,a.bank,a.verifier,HybridSettings(cnn_threshold=a.cnn_threshold,feature_upsample=a.feature_upsample,feature_ransac_iterations=a.feature_ransac_iterations))
    bank=json.loads((a.bank/'manifest.json').read_text());fixture=json.loads((a.fixture/'fixture.json').read_text())
    report=dict(initialization_seconds=time.perf_counter()-start,checkpoint_sha256=sha(a.weights),bank_sha256=sha(a.bank/'manifest.json'),
                verifier_sha256=sha(a.verifier),fixture_sha256=sha(a.fixture/'fixture.json'),cnn_threshold=a.cnn_threshold,background_threshold=.5,
                feature_threshold=.8,feature_upsample=a.feature_upsample,feature_ransac_iterations=a.feature_ransac_iterations,zoom=2,results=[],versions=dict(torch=torch.__version__,opencv=cv2.__version__),
                code_hashes={str(path):sha(path) for path in [Path(__file__),Path(__file__).with_name('hybrid.py'),Path(__file__).with_name('patch_cnn.py'),Path(__file__).parents[1]/'template_matching/features.py']})
    for source in fixture['examples']:
        if source['sha256'] in bank['source_hashes'].values():raise ValueError('Training source overlaps evaluation')
        if source['split']=='validation' and source['frame']<=bank['validation_train_through']:raise ValueError('Training interval overlap')
        if any(t.get('track') in bank['validation_train_tracks'] for t in source['truth']):raise ValueError('Training track overlap')
        path=a.fixture/source['file']
        if sha(path)!=source['sha256']:raise ValueError('Image hash mismatch')
        image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if image.shape[2]==4 and np.any(image[:,:,3]!=255):raise ValueError('Unobserved pixels')
        image=image[:,:,:3];predictions=[];timings=[]
        for y in starts(2160,540):
            for x in starts(3840,960):
                start=time.perf_counter();rows=detector.detect(image[y:y+540,x:x+960],1.);timings.append(time.perf_counter()-start)
                for r in rows:
                    b=r['bbox'];predictions.append({**r,'bbox':[x+b[0],y+b[1],x+b[2],y+b[3]]})
        predictions=detector.suppress(predictions)
        truth=[t for t in source['truth'] if 0<t['bbox'][0]<t['bbox'][2]<3839 and 0<t['bbox'][1]<t['bbox'][3]<2159]
        row=dict(split=source['split'],frame=source['frame'],image_sha256=source['sha256'],predictions=predictions,truth=truth,
                 view_seconds=timings,seconds=sum(timings),**measure(predictions,truth))
        report['results'].append(row);(a.output/'report.json').write_text(json.dumps(report,indent=2))
        overlay=cv2.resize(image,(1920,1080))
        for t in truth:
            x1,y1,x2,y2=[round(v/2) for v in t['bbox']];cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,180,0),1)
        for r in predictions:
            x1,y1,x2,y2=[round(v/2) for v in r['bbox']];color=(0,170,255) if r.get('method')=='features' else (0,255,0)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,1);cv2.putText(overlay,f"{r['class']} {r['score']:.2f}",(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
        cv2.imwrite(str(a.output/f"{source['split']}-{source['frame']:06d}.jpg"),overlay)
        print(json.dumps({k:row[k] for k in ['split','frame','seconds','targets','matched','proposals']}),flush=True)


if __name__=='__main__':main()
