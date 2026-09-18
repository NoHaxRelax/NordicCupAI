import argparse,json,time
from pathlib import Path
import cv2,torch
from .detector import AssetHeatmapDetector
from .data import sha
from drone.scratch_objects.evaluate import starts,measure,nms


def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(2)
 model=AssetHeatmapDetector(a.weights,.02,a.device);fixture=json.loads((a.fixture/'fixture.json').read_text());report=dict(weights_sha256=sha(a.weights),fixture_sha256=sha(a.fixture/'fixture.json'),results=[])
 for e in fixture['examples']:
  path=a.fixture/e['file'];assert sha(path)==e['sha256'];im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
  if im.shape[2]==4 and (im[:,:,3]!=255).any():raise ValueError('Unobserved pixels')
  im=im[:,:,:3];rows=[];start=time.perf_counter()
  for y in starts(2160,540):
   for x in starts(3840,960):
    for r in model.detect(im[y:y+540,x:x+960]):
     b=r['bbox'];rows.append({**r,'bbox':[x+b[0],y+b[1],x+b[2],y+b[3]]})
  rows=nms(rows);truth=[t for t in e['truth'] if 0<t['bbox'][0]<t['bbox'][2]<3839 and 0<t['bbox'][1]<t['bbox'][3]<2159]
  r=dict(split=e['split'],frame=e['frame'],truth=truth,predictions=rows,seconds=time.perf_counter()-start,thresholds={str(t):measure([p for p in rows if p['score']>=t],truth)for t in [.05,.1,.2,.3,.5,.7]});report['results'].append(r);(a.output/'report.json').write_text(json.dumps(report,indent=2));print(e['frame'],{t:(v['matched'],v['proposals'])for t,v in r['thresholds'].items()},flush=True)


if __name__=='__main__':main()
