"""First-run training. Manual positives are never used as detector negatives."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def runtime():
    import importlib.metadata
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_properties(0)
    return dict(gpu=gpu.name, vram_gib=gpu.total_memory/2**30, torch=torch.__version__,
                cuda=torch.version.cuda, ultralytics=importlib.metadata.version('ultralytics'),
                job=os.environ.get('LSB_JOBID'), commit=os.environ.get('DRONE_COMMIT'),
                hostname=platform.node(), platform=platform.platform())


def detector(args):
    from ultralytics import YOLO
    model = YOLO(str(args.weights / 'yolo26x.pt'))
    # Absolute dataset root avoids Ultralytics global datasets_dir surprises.
    data = args.output / 'detector-data.yaml'
    data.write_text(f'path: {args.data.resolve() / "detector"}\ntrain: images/train\nval: images/train\nnames: '+json.dumps(args.manifest['classes'])+'\n')
    def progress(trainer):
        write(args.output/'progress.json', dict(task='detector', epoch=trainer.epoch+1,
              total_epochs=args.epochs, loss=trainer.tloss.detach().cpu().tolist(),
              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              warning='Any detector validation metrics are training-fit diagnostics, not holdout performance.'))
    model.add_callback('on_train_epoch_end', progress)
    model.train(data=str(data), epochs=args.epochs, imgsz=960, batch=args.batch, device=0,
                project=str(args.output), name='detector', exist_ok=False,
                pretrained=True, amp=True, workers=args.workers, seed=170926, deterministic=True,
                optimizer='AdamW', lr0=0.001, lrf=0.05, warmup_epochs=3,
                mosaic=0.0, mixup=0.0, degrees=5.0, translate=0.05, scale=0.1,
                fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.2, hsv_v=0.2,
                patience=0, val=False, plots=False, save=True, save_period=5, nbs=64,
                cache=False, close_mosaic=0)
    checkpoint = args.output/'detector/weights/last.pt'
    write(args.output/'result.json', dict(task='detector', status='completed', epochs=args.epochs,
          checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
          selection='fixed final epoch; training-fit best.pt is not used',
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


class Crops(torch.utils.data.Dataset):
    def __init__(self, root, rows, classes, augment=False):
        self.root, self.rows, self.classes, self.augment = root, rows, classes, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image, ImageOps
        from torchvision.transforms import functional as F
        r = self.rows[index]
        im = Image.open(self.root/r['file']).convert('RGB')
        # The source has already been rendered at the real camera resolution.
        # Resizing here cannot restore information lost at L0/L1.
        im = ImageOps.pad(im, (224, 224), method=Image.Resampling.BILINEAR, color=(114,114,114))
        if self.augment and random.random() < .5:
            im = ImageOps.mirror(im)
        tensor = F.normalize(F.to_tensor(im), [.485,.456,.406], [.229,.224,.225])
        return tensor, self.classes.index(r['class_name']), r['zoom']


@torch.inference_mode()
def evaluate(model, loader, classes):
    model.eval()
    matrices = {z:np.zeros((len(classes),len(classes)), dtype=int) for z in range(3)}
    for x, y, zoom in loader:
        with torch.autocast('cuda', dtype=torch.float16):
            pred = model(x.cuda(non_blocking=True)).argmax(1).cpu()
        for target, guess, z in zip(y.tolist(), pred.tolist(), zoom.tolist()):
            matrices[z][target, guess] += 1
    report = {}
    for z, cm in matrices.items():
        counts = cm.sum(1)
        per_class = {c:dict(n=int(counts[i]), accuracy=float(cm[i,i]/counts[i]))
                     for i,c in enumerate(classes) if counts[i]}
        report[f'L{z}'] = dict(n=int(cm.sum()), accuracy=float(np.trace(cm)/cm.sum()),
                              macro_accuracy=float(np.mean([r['accuracy'] for r in per_class.values()])),
                              per_class=per_class, confusion_matrix=cm.tolist())
    return report


def classifier(args):
    from torchvision.models import resnet50, ResNet50_Weights
    classes = args.manifest['classes']
    rows = [r for r in args.manifest['records'] if r['task']=='classifier']
    tr, va = ([r for r in rows if r['split']==s] for s in ('train','val'))
    counts = Counter((r['class_name'], r['zoom']) for r in tr)
    weights = [{0:.1,1:.7,2:.2}[r['zoom']]/counts[r['class_name'],r['zoom']] for r in tr]
    gen = torch.Generator().manual_seed(170926)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(tr), replacement=True, generator=gen)
    train_loader = torch.utils.data.DataLoader(Crops(args.data,tr,classes,True), batch_size=args.batch,
                   sampler=sampler, num_workers=args.workers, pin_memory=True, generator=gen)
    val_loader = torch.utils.data.DataLoader(Crops(args.data,va,classes), batch_size=args.batch,
                 shuffle=False, num_workers=args.workers, pin_memory=True)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, count, start = 0., 0, time.perf_counter()
        for x,y,_ in train_loader:
            x,y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x),y,label_smoothing=.05)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()*len(x)
            count += len(x)
        scheduler.step()
        metrics = evaluate(model,val_loader,classes)
        row = dict(epoch=epoch+1, train_loss=total_loss/count, holdout=metrics,
                   seconds=time.perf_counter()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        history.append(row)
        write(args.output/'progress.json', row)
        write(args.output/'history.json', history)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), epoch=epoch+1, classes=classes, architecture='resnet50',
                       manifest_sha256=sha(args.data/'manifest.json'), seed=170926,
                       rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                                numpy=np.random.get_state(), python=random.getstate(), sampler=gen.get_state()))
        temporary = args.output/'last.tmp'
        torch.save(payload,temporary)
        temporary.replace(args.output/'last.pt')
        print(json.dumps(dict(epoch=epoch+1, train_loss=row['train_loss'], L1_macro=metrics['L1']['macro_accuracy'], seconds=row['seconds'])), flush=True)
    checkpoint = args.output/'last.pt'
    write(args.output/'result.json', dict(task='classifier', status='completed', epochs=args.epochs,
          checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), holdout=history[-1]['holdout'],
          selection='fixed final epoch; no hyperparameter search on holdout',
          limits=args.manifest['limits'], peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task', choices=['detector','classifier'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch',type=int,help='Defaults: detector 8, classifier 64. Detector nominal batch remains 64 via gradient accumulation.')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.batch is None:
        args.batch = 8 if args.task == 'detector' else 64
    if args.batch < 1 or args.workers < 0:
        p.error('batch must be positive and workers nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    args.manifest=json.loads((args.data/'manifest.json').read_text())
    random.seed(170926); np.random.seed(170926); torch.manual_seed(170926); torch.cuda.manual_seed_all(170926)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    write(args.output/'runtime.json',dict(runtime(), task=args.task, epochs=args.epochs, batch=args.batch, workers=args.workers,
          data_manifest_sha256=sha(args.data/'manifest.json'), started_at=time.time()))
    {'detector':detector,'classifier':classifier}[args.task](args)


if __name__=='__main__':
    main()
