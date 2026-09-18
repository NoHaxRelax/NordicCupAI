"""Standalone CUDA training entry point. No pretrained parameters are loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=96)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--degrees',type=float,default=0.)
    p.add_argument('--model',default='yolo26m.yaml')
    p.add_argument('--amp',action='store_true')
    p.add_argument('--continue-from',type=Path)
    p.add_argument('--origin-completion',type=Path)
    a=p.parse_args()
    if a.imgsz<32 or a.imgsz%32 or a.lr<=0 or not 0<=a.degrees<=180:
        raise ValueError('Invalid image size, learning rate or rotation')
    if not a.model.endswith('.yaml'):raise ValueError('Architecture YAML required for scratch initialization')
    if a.continue_from:
        if not a.origin_completion or json.loads(a.origin_completion.read_text())['last_sha256']!=digest(a.continue_from):
            raise ValueError('Continuation requires matching completion receipt for the prior local scratch run')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch
    import ultralytics
    from ultralytics import YOLO,settings
    if a.amp:
        # Upstream AMP checking loads pretrained weights only to compare arithmetic.
        # Our CUDA run instead verifies finite training losses and uses no such model.
        import ultralytics.engine.trainer as trainer_module
        trainer_module.check_amp=lambda model: True
    settings.update({'wandb':False,'mlflow':False,'clearml':False,'comet':False})
    if not torch.cuda.is_available():raise RuntimeError('Verified CUDA device required')
    torch.set_num_threads(4)
    manifest=json.loads((a.data/'manifest.json').read_text())
    for row in manifest['records']:
        for key,hashkey in [('file','sha256'),('label','label_sha256')]:
            if digest(a.data/row[key])!=row[hashkey]:raise ValueError('Dataset hash mismatch '+row[key])
    dataset=a.output/'dataset.yaml'
    dataset.write_text(json.dumps(dict(path=str(a.data.resolve()),train='images/train',val='images/synthetic_dev',names=manifest['classes'])))
    receipt=dict(amp=a.amp,initialization='random, architecture YAML only',architecture=a.model,epochs=a.epochs,batch=a.batch,
                 imgsz=a.imgsz,lr=a.lr,degrees=a.degrees,
                 data_sha256=digest(a.data/'manifest.json'),code_sha256=digest(__file__),
                 torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0),
                 validation='Synthetic fitting diagnostic only; final checkpoint used for real transfer evaluation')
    if a.continue_from:
        receipt.update(initialization='continued from local scratch experiment',origin_checkpoint_sha256=digest(a.continue_from),origin_completion_sha256=digest(a.origin_completion))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    import random
    import numpy as np
    random.seed(1731);np.random.seed(1731);torch.manual_seed(1731);torch.cuda.manual_seed_all(1731)
    model=YOLO(str(a.continue_from) if a.continue_from else a.model)
    expected_parameters={name:p.detach().float().cpu().clone() for name,p in model.model.named_parameters()} if a.continue_from else None
    def verify_continuation(trainer):
        if expected_parameters is None:return
        actual=dict(trainer.model.named_parameters())
        mismatched=[name for name,p in expected_parameters.items() if name not in actual or not torch.equal(p,actual[name].detach().float().cpu())]
        if mismatched:raise RuntimeError('Continuation discarded or changed loaded parameters: '+str(mismatched[:5]))
        (a.output/'continuation-verified.json').write_text(json.dumps(dict(parameters_verified=len(expected_parameters),origin_checkpoint_sha256=digest(a.continue_from))))
        expected_parameters.clear()
    model.add_callback('on_train_start',verify_continuation)
    started=time.time()
    def progress(trainer):
        values=trainer.tloss
        if isinstance(values,dict):
            loss={k:float(v.detach().mean().cpu()) if torch.is_tensor(v) else float(v) for k,v in values.items()}
            finite=all(__import__('math').isfinite(v) for v in loss.values())
        else:
            loss=values.detach().cpu().tolist();finite=bool(torch.isfinite(values).all())
        if not finite:raise RuntimeError('Nonfinite training loss')
        row=dict(epoch=trainer.epoch+1,epochs=a.epochs,seconds=time.time()-started,loss=loss,
                 peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        (a.output/'progress.json').write_text(json.dumps(row,indent=2))
    model.add_callback('on_train_epoch_end',progress)
    # Disable AMP's optional pretrained-model comparison/download; this run is wholly scratch.
    model.train(data=str(dataset.resolve()),epochs=a.epochs,batch=a.batch,imgsz=a.imgsz,device=0,workers=0,
                project=str(a.output.resolve()),name='fit',exist_ok=False,pretrained=bool(a.continue_from),amp=a.amp,
                optimizer='AdamW',lr0=a.lr,lrf=.05,weight_decay=.0005,warmup_epochs=3,nbs=64,
                seed=1731,deterministic=True,mosaic=0,mixup=0,degrees=a.degrees,scale=.2,translate=.1,
                fliplr=.5,flipud=.5,hsv_h=.01,hsv_s=.15,hsv_v=.15,patience=0,close_mosaic=0,
                plots=False,save=True,save_period=24,cache=False)
    (a.output/'complete.json').write_text(json.dumps(dict(seconds=time.time()-started,last_sha256=digest(a.output/'fit/weights/last.pt'))))


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        Path(__file__).with_suffix('.failure.txt').write_text(traceback.format_exc())
        raise
