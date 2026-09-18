"""One experiment per process. Existing outputs are never silently overwritten."""
import argparse
from collections import Counter
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

from common import fingerprint, read, release_errors, sha, verify_dataset, write


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def detector(args, spec, manifest):
    import torch
    from ultralytics import YOLO, settings
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from copy import copy
    from single_label import SingleLabelValidator
    class AP50Trainer(DetectionTrainer):
        def get_validator(self):
            return SingleLabelValidator(self.test_loader, save_dir=self.save_dir,
                                        args=copy(self.args), _callbacks=self.callbacks)

        def validate(self):
            previous=self.best_fitness
            metrics,fitness=super().validate()
            if metrics is None:return metrics,fitness
            score=float(metrics['metrics/mAP50(B)'])
            self.best_fitness=score if previous is None else max(previous,score)
            return metrics,score
    settings.update({'wandb':False, 'mlflow':False, 'clearml':False, 'comet':False})
    architecture=spec.get('detector_model','yolo26x')
    if architecture not in ['yolo26x','yolo26m']:raise ValueError('Unknown detector architecture')
    if args.resume:
        model=YOLO(str(args.output/'fit/weights/last.pt'))
    elif spec['initialization']=='scratch':
        model=YOLO(architecture+'.yaml')
    else:
        model=YOLO(str(args.weights/(architecture+'.pt')))
    def data_yaml(zoom, suffix):
        paths={}
        for split in ['train','dev']:
            rows=[r for r in manifest['records'] if r['task']=='detector' and r['split']==split
                  and (zoom=='mixed' or r['zoom']==zoom)]
            if not rows:
                raise ValueError(f'No {split} detector views at zoom {zoom}')
            listing=args.output/f'{split}-{suffix}.txt'
            listing.write_text(''.join(str((args.data/r['file']).resolve())+'\n' for r in rows))
            paths[split]=str(listing.resolve())
        path=args.output/f'data-{suffix}.yaml'
        # JSON is a YAML subset; preserves Windows paths without escaping errors.
        write(path,dict(path=str(args.data.resolve()),train=paths['train'],val=paths['dev'],names=manifest['classes']))
        return str(path.resolve())
    data=data_yaml(spec['zoom'],'training')
    modules=model.model.model
    freeze={'full':[], 'head':list(range(len(modules)-1)),
            'partial':list(range(len(model.model.yaml['backbone'])))}[spec['adaptation']]
    start=[time.perf_counter()]
    def epoch_start(trainer):
        start[0]=time.perf_counter()
        # Frozen weights must also retain frozen batch-normalisation statistics.
        for index in freeze:
            trainer.model.model[index].eval()
    def progress(trainer):
        values=trainer.tloss
        if isinstance(values,dict):
            loss={k:float(v.detach().mean().cpu()) if torch.is_tensor(v) else float(v) for k,v in values.items()}
        else:
            loss=values.detach().cpu().tolist()
        row=dict(state='running',epoch=trainer.epoch+1,total_epochs=spec['epochs'],loss=loss,
                 updated_at=now(),seconds=time.perf_counter()-start[0],
                 peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        write(args.output/'progress.json',row)
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_train_epoch_end',progress)
    opts=dict(data=data,epochs=spec['epochs'],imgsz=960,batch=args.batch,workers=args.workers,device=0,
        project=str(args.output.resolve()),name='fit',exist_ok=args.resume,pretrained=spec['initialization']=='pretrained',
        amp=True,seed=manifest['config']['seed'],deterministic=True,optimizer='AdamW',lr0=spec['lr'],
        lrf=0.05,warmup_epochs=3,freeze=freeze,mosaic=0.0,mixup=0.0,degrees=180.0,translate=0.1,
        scale=0.3,fliplr=0.5,flipud=0.5,hsv_h=0.01,hsv_s=0.2,hsv_v=0.2,patience=0,
        val=True,plots=False,save=True,save_period=10,nbs=64,cache=False,close_mosaic=0,resume=args.resume)
    write(args.output/'training-options.json',opts)
    model.train(trainer=AP50Trainer,**opts)
    checkpoint=args.output/'fit/weights/best.pt'
    selected=YOLO(str(checkpoint))
    metrics={}
    # All zooms are evaluated for every specialist as well as the mixed model.
    for z in range(3):
        result=selected.val(validator=SingleLabelValidator,data=data_yaml(z,f'eval-L{z}'),imgsz=960,batch=args.batch,device=0,
                            workers=args.workers,plots=False,verbose=False,
                            project=str(args.output.resolve()),name=f'eval-L{z}')
        metrics[f'L{z}']=dict(map50=float(result.box.map50),map50_95=float(result.box.map),
            represented_classes=[manifest['classes'][int(i)] for i in result.box.ap_class_index],
            per_class_ap50={manifest['classes'][int(i)]:float(ap) for i,ap in zip(result.box.ap_class_index,result.box.ap50)})
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),dev=metrics,
                validation_protocol='single-label NMS, matching DetectionPredictor class selection',
                selection='best development AP50; AP50 trainer explicitly replaces default fitness',
                limits=manifest['limitations'])


def classifier(args,spec,manifest):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision.models import resnet50
    from torchvision.transforms import functional as F
    from PIL import Image, ImageOps
    classes=manifest['classifier_classes']
    input_size=int(spec.get('input_size',224))
    if input_size not in [224,384]:raise ValueError('Unsupported classifier input size')
    # This dataset lives inside the function, so Windows uses num_workers=0 for
    # classifier loading. Detector loading still uses its tested worker setting.
    class Crops(Dataset):
        def __init__(self,rows,augment=False):
            self.rows,self.augment=rows,augment
            # Small decoded crops fit comfortably in host RAM. Read each only
            # once, avoiding thousands of persistent-volume opens each epoch.
            self.images=[]
            for row in rows:
                with Image.open(args.data/row['file']) as source:
                    self.images.append(source.convert('RGB'))
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            row=self.rows[index];im=self.images[index].copy()
            im=ImageOps.pad(im,(input_size,input_size),method=Image.Resampling.BILINEAR,color=(114,114,114))
            if self.augment:
                mode=spec.get('rotation_mode','original')
                if mode not in ['original','expanded','none']:raise ValueError('Unknown rotation mode')
                if mode!='none':
                    im=im.rotate(random.uniform(-180,180),resample=Image.Resampling.BILINEAR,fillcolor=(114,114,114),expand=mode=='expanded')
                    if mode=='expanded':im=ImageOps.pad(im,(input_size,input_size),method=Image.Resampling.BILINEAR,color=(114,114,114))
                if random.random()<0.5: im=ImageOps.mirror(im)
            return F.normalize(F.to_tensor(im),[.485,.456,.406],[.229,.224,.225]),classes.index(row['class_name']),row['zoom']
    tr=[r for r in manifest['records'] if r['task']=='classifier' and r['split']=='train'
        and (spec['zoom']=='mixed' or r['zoom']==spec['zoom'])
        and (spec.get('train_source','mixed')=='mixed' or r['source']==spec['train_source'])
        and (r['class_name'] not in spec.get('reference_only_classes',[]) or r['source']=='reference')]
    va=[r for r in manifest['records'] if r['task']=='classifier' and r['split']=='dev']
    counts=Counter((r['class_name'],r['zoom']) for r in tr)
    generator=torch.Generator().manual_seed(manifest['config']['seed'])
    sampler=WeightedRandomSampler([1/counts[r['class_name'],r['zoom']] for r in tr],len(tr),replacement=True,generator=generator)
    loader=DataLoader(Crops(tr,True),batch_size=args.batch,sampler=sampler,num_workers=0,pin_memory=True)
    validation=DataLoader(Crops(va),batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=True)
    architecture=spec.get('architecture','resnet50')
    if architecture in ['convnext_tiny','convnext_small']:
        from torchvision.models import convnext_tiny,convnext_small
        if spec['adaptation']!='full':raise ValueError('ConvNeXt experiment supports full adaptation only')
        model={'convnext_tiny':convnext_tiny,'convnext_small':convnext_small}[architecture](weights=None)
        if spec['initialization']=='pretrained' and not args.resume:
            model.load_state_dict(torch.load(args.weights/({'convnext_tiny':'convnext_tiny-983f1562.pth','convnext_small':'convnext_small-0c510722.pth'}[architecture]),map_location='cpu',weights_only=True))
        model.classifier[2]=torch.nn.Linear(model.classifier[2].in_features,len(classes))
    elif architecture=='resnet50':
        model=resnet50(weights=None)
        if spec['initialization']=='pretrained' and not args.resume:
            model.load_state_dict(torch.load(args.weights/'resnet50-11ad3fa6.pth',map_location='cpu',weights_only=True))
        model.fc=torch.nn.Linear(model.fc.in_features,len(classes))
    else:raise ValueError('Unknown classifier architecture')
    if spec['adaptation']!='full':
        for name,param in model.named_parameters():
            param.requires_grad=name.startswith('fc.') or (spec['adaptation']=='partial' and name.startswith('layer4.'))
    model.cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec['lr'],weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,spec['epochs'])
    scaler=torch.amp.GradScaler('cuda')
    epoch0,best=0,-1.
    if args.resume:
        checkpoint=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        epoch0,best=checkpoint['epoch'],checkpoint['best']
        random.setstate(checkpoint['rng']['python']);np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch']);torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        generator.set_state(checkpoint['rng']['sampler'])
    @torch.inference_mode()
    def evaluate():
        model.eval();matrices={z:np.zeros((len(classes),len(classes)),dtype=int) for z in range(3)}
        for x,y,z in validation:
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x.cuda()).argmax(1).cpu()
            for truth,guess,zoom in zip(y.tolist(),pred.tolist(),z.tolist()):matrices[zoom][truth,guess]+=1
        report={}
        for z,cm in matrices.items():
            n=cm.sum(1);present=n>0
            foreground=present.copy();foreground[-1]=False
            report[f'L{z}']=dict(n=int(cm.sum()),accuracy=float(np.trace(cm)/cm.sum()) if cm.sum() else None,
                foreground_macro_accuracy=float(np.mean(cm.diagonal()[foreground]/n[foreground])) if foreground.any() else None,
                macro_accuracy=float(np.mean(cm.diagonal()[present]/n[present])) if present.any() else None,
                background_false_positive_rate=float(1-cm[-1,-1]/n[-1]) if n[-1] else None,
                per_class={c:dict(n=int(n[i]),recall=float(cm[i,i]/n[i])) for i,c in enumerate(classes) if n[i]},
                confusion_matrix=cm.tolist())
        return report
    best_metrics=None
    for epoch in range(epoch0,spec['epochs']):
        model.train()
        if spec['adaptation']!='full':
            # Keep frozen backbone statistics fixed; only layer4 trains in partial mode.
            for name,module in model.named_children():
                if name!='fc' and not(spec['adaptation']=='partial' and name=='layer4'):module.eval()
        total,n,start=0.,0,time.perf_counter()
        for x,y,_ in loader:
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(x),y,label_smoothing=.05)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total+=float(loss)*len(x);n+=len(x)
        scheduler.step();metrics=evaluate()
        selection_metrics=metrics.values() if spec['zoom']=='mixed' else [metrics[f'L{spec["zoom"]}']]
        score=float(np.mean([m['macro_accuracy'] for m in selection_metrics if m['macro_accuracy'] is not None]))
        improved=score>best
        if improved:best=score;best_metrics=metrics
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch+1,best=best,classes=classes,spec=spec,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all(),sampler=generator.get_state()))
        torch.save(payload,args.output/'last.tmp');(args.output/'last.tmp').replace(args.output/'last.pt')
        if improved:
            torch.save(dict(model=model.state_dict(),classes=classes,spec=spec,epoch=epoch+1),args.output/'best.tmp')
            (args.output/'best.tmp').replace(args.output/'best.pt');write(args.output/'best-metrics.json',metrics)
        row=dict(state='running',epoch=epoch+1,total_epochs=spec['epochs'],train_loss=total/n,dev=metrics,
                 seconds=time.perf_counter()-start,updated_at=now(),peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        write(args.output/'progress.json',row)
        with (args.output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(dict(epoch=epoch+1,train_loss=total/n,dev_macro=score,seconds=row['seconds'])),flush=True)
    return dict(checkpoint=str(args.output/'best.pt'),checkpoint_sha256=sha(args.output/'best.pt'),
                dev=read(args.output/'best-metrics.json'),selection=f'best development macro accuracy including background; selection zoom={spec["zoom"]}',
                limits=manifest['limitations'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','weights','output','release']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--experiment',required=True);p.add_argument('--host',choices=['mypc','hpc','runpod'],required=True)
    p.add_argument('--batch',type=int);p.add_argument('--workers',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();manifest=read(args.data/'manifest.json');release=read(args.release)
    errors=release_errors(manifest,release)
    if errors:raise SystemExit('; '.join(errors))
    spec=next((e.copy() for e in manifest['config']['experiments'] if e['id']==args.experiment),None)
    if spec is None:raise SystemExit('Unknown experiment')
    if spec['task']=='detector':spec['validation_protocol']='single-label-nms-v1'
    if args.smoke:spec['epochs']=1
    host=manifest['config']['hosts'][args.host]
    args.batch=args.batch or host[spec['task']+'_batch'];args.workers=host['workers'] if args.workers is None else args.workers
    if args.batch<1 or args.workers<0:raise SystemExit('Invalid batch/workers')
    # Fail before allocating a GPU if a supposedly frozen dataset changed.
    verify_dataset(args.data,manifest)
    identity=dict(manifest_sha256=sha(args.data/'manifest.json'),spec=spec,host=args.host,batch=args.batch,
                  code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    if args.resume:
        previous=read(args.output/'runtime.json')
        if previous['identity']!=identity:raise SystemExit('Resume requires identical code, data, options and host')
        if (args.output/'result.json').exists():raise SystemExit('Run already completed')
    else:args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise SystemExit('Require exactly one visible CUDA GPU; respect scheduler mask')
    seed=manifest['config']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    write(args.output/'runtime.json',dict(identity=identity,started_at=now(),host=platform.node(),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        ultralytics=importlib.metadata.version('ultralytics'),pid=os.getpid()))
    try:
        result={'detector':detector,'classifier':classifier}[spec['task']](args,spec,manifest)
        result.update(state='completed',finished_at=now(),experiment=spec['id'],smoke=args.smoke)
        write(args.output/'result.json',result)
        write(args.output/'progress.json',dict(state='completed',updated_at=now()))
    except BaseException as exc:
        write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),updated_at=now()))
        raise


if __name__=='__main__':main()
