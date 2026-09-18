"""Blind native-view evaluation of the fixed-asset runtime on a frozen fixture."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import cv2,torch
from .fixed_assets import FixedAssetDetector,FixedAssetSettings
from .evaluate import starts,measure
from .prepare import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--cnn-threshold',type=float,default=.6);p.add_argument('--no-geometric',action='store_true');p.add_argument('--verifier',type=Path);p.add_argument('--heatmap-weights',type=Path);p.add_argument('--disable-cnn',action='store_true');p.add_argument('--all-heatmap-classes',action='store_true');p.add_argument('--heatmap-threshold',type=float,default=.3);p.add_argument('--tiny-min-contrast',type=float,default=.2);p.add_argument('--verify-heatmap',action='store_true');p.add_argument('--heatmap-verifier-threshold',type=float,default=.7);p.add_argument('--calibrated-pixels',action='store_true');p.add_argument('--calibrated-pixel-threshold',type=float,default=.7);p.add_argument('--disable-heatmap',action='store_true');p.add_argument('--verified-feature-proposals',action='store_true');p.add_argument('--verify-tiny',action='store_true');p.add_argument('--cnn-verifier-threshold',type=float,default=.7);p.add_argument('--heatmap-class-gate',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);cv2.setNumThreads(4);a.output.mkdir(parents=True,exist_ok=False)
 cfg=FixedAssetSettings(heatmap_reclassify=not a.heatmap_class_gate,cnn_verifier_threshold=a.cnn_verifier_threshold,heatmap_enabled=not a.disable_heatmap,feature_threshold=.7 if a.verified_feature_proposals else .8,verify_features_below=.8 if a.verified_feature_proposals else 0.,verify_tiny=a.verify_tiny,calibrated_pixels=a.calibrated_pixels,calibrated_pixel_threshold=a.calibrated_pixel_threshold,cnn_threshold=a.cnn_threshold,geometric=not a.no_geometric,device=a.device,cnn_enabled=not a.disable_cnn,heatmap_classes=() if a.all_heatmap_classes else ('small_tower',),heatmap_threshold=a.heatmap_threshold,tiny_min_contrast=a.tiny_min_contrast,verify_heatmap=a.verify_heatmap,heatmap_verifier_threshold=a.heatmap_verifier_threshold);start=time.perf_counter();model=FixedAssetDetector(a.weights,a.bank,cfg,a.verifier,a.heatmap_weights)
 fixture=json.loads((a.fixture/'fixture.json').read_text());report=dict(settings=asdict(cfg),initialization_seconds=time.perf_counter()-start,checkpoint_sha256=sha(a.weights),bank_sha256=sha(a.bank/'manifest.json'),fixture_sha256=sha(a.fixture/'fixture.json'),verifier_sha256=sha(a.verifier) if a.verifier else None,heatmap_sha256=sha(a.heatmap_weights) if a.heatmap_weights else None,pose_calibration=json.loads((a.bank/'manifest.json').read_text()).get('pose_calibration'),results=[])
 for e in fixture['examples']:
  path=a.fixture/e['file'];assert sha(path)==e['sha256'];im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
  if im.shape[2]==4 and (im[:,:,3]!=255).any():raise ValueError('Unobserved pixels')
  im=im[:,:,:3];rows=[];timings=[]
  for y in starts(2160,540):
   for x in starts(3840,960):
    start=time.perf_counter();predictions=model.detect(im[y:y+540,x:x+960]);timings.append(time.perf_counter()-start)
    for r in predictions:
     b=r['bbox'];rows.append({**r,'bbox':[x+b[0],y+b[1],x+b[2],y+b[3]]})
  rows=model.suppress(rows);truth=[t for t in e['truth'] if 0<t['bbox'][0]<t['bbox'][2]<3839 and 0<t['bbox'][1]<t['bbox'][3]<2159]
  result=dict(split=e['split'],frame=e['frame'],truth=truth,predictions=rows,view_seconds=timings,**measure(rows,truth));report['results'].append(result);(a.output/'report.json').write_text(json.dumps(report,indent=2))
  print(json.dumps({k:result[k] for k in ['split','frame','matched','targets','proposals']}),flush=True)
  overlay=cv2.resize(im,(1920,1080))
  for t in truth:
   x1,y1,x2,y2=[round(v/2) for v in t['bbox']];cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,180,0),1)
  for r in rows:
   x1,y1,x2,y2=[round(v/2) for v in r['bbox']];color=(0,255,0) if r['family']=='cnn' else (0,170,255);cv2.rectangle(overlay,(x1,y1),(x2,y2),color,1);cv2.putText(overlay,r['class'],(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
  cv2.imwrite(str(a.output/f"{e['split']}-{e['frame']:06d}.jpg"),overlay)


if __name__=='__main__':main()
