"""Short batch-one GPU check; timings exclude network and the serving adapter."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from train import runtime, write

p=argparse.ArgumentParser()
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,help='Optional trained checkpoint; otherwise use pretrained yolo26x.pt')
p.add_argument('--shared-gpu',action='store_true')
args=p.parse_args()
torch.set_num_threads(4)
info=runtime()
paths=sorted((args.data/'detector/images/train').glob('*.png'))[:10]
encoded=[p.read_bytes() for p in paths]
checkpoint=args.checkpoint or args.weights/'yolo26x.pt'
model=YOLO(str(checkpoint))
samples=[]
for i in range(70):
    torch.cuda.synchronize()
    start=time.perf_counter()
    img=cv2.imdecode(np.frombuffer(encoded[i%len(encoded)],dtype=np.uint8),cv2.IMREAD_COLOR)
    result=model.predict(img,imgsz=960,device=0,half=True,verbose=False)
    boxes=result[0].boxes.data.cpu().tolist()
    json.dumps(boxes)
    torch.cuda.synchronize()
    if i>=20:
        samples.append((time.perf_counter()-start)*1000)
assert model.predictor.device.type == 'cuda'
info.update(samples=len(samples), batch=1, delivered_size=[960,540], checkpoint=str(checkpoint),
            p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),
            p99_ms=float(np.percentile(samples,99)),max_ms=max(samples),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            scope='PNG decode, preprocessing, GPU inference, postprocessing, CPU boxes and JSON; no HTTP, network, camera or tracker.',
            shared_gpu=args.shared_gpu, limits='GPU smoke only; repeat on final fine-tuned artifact and actual endpoint before claiming the 333 ms serving target.')
write(args.output,info)
print(json.dumps(info,indent=2))
