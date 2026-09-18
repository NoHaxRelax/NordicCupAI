"""Train the candidate verifier: presence and class of an expert candidate crop.

A pretrained ResNet-18 (torchvision weights, allowed by the rules) fine-tuned on 96 px crops with a
17-way head (16 classes + background). Inputs: harvested expert crops (real training tiles) and,
optionally, synthetic sprite composites cut the same way. Mild augmentation only: the assets look the
same every time, so flips, quarter turns, small scale jitter and modest brightness/contrast changes.
Holds out 15% of the real crops by tile id for a fitting check. No dev or reserved data is read.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASSES = ['background', 'condor', 'hangar', 'helicopter', 'jammer', 'jet_plane', 'large_launcher', 'large_tower', 'medium_launcher',
           'medium_plane', 'mine_roller', 'small_launcher', 'small_plane', 'small_tower', 'spacecraft', 'ta-ta', 'tank']


class Crops(Dataset):
    def __init__(self, images, labels, train):
        self.images, self.labels, self.train = images, labels, train
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i][:, :, ::-1].astype(np.float32) / 255.  # BGR -> RGB
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < .5:
                img = img[:, ::-1]
            img = np.rot90(img, int(rng.integers(4)))
            img = np.clip(img * rng.uniform(.85, 1.15) + rng.uniform(-.05, .05), 0, 1)
        x = (img.transpose(2, 0, 1) - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)).float(), int(self.labels[i])


def load(path):
    doc = json.loads((Path(path) / 'manifest.json').read_text())
    data = np.load(Path(path) / 'crops.npz')
    labels = np.array([CLASSES.index(l) for l in data['labels']])
    tiles = np.array([r['tile'] for r in doc['records']]) if 'records' in doc and len(doc['records']) == len(labels) else np.array([str(i) for i in range(len(labels))])
    return data['images'], labels, tiles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--crops', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--holdout', type=float, default=.15)
    p.add_argument('--seed', type=int, default=1731)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    images, labels, tiles = zip(*[load(c) for c in a.crops])
    images, labels, tiles = np.concatenate(images), np.concatenate(labels), np.concatenate(tiles)
    rng = np.random.default_rng(a.seed)
    unique = np.array(sorted(set(tiles)))
    held = set(rng.choice(unique, size=max(1, int(a.holdout * len(unique))), replace=False).tolist())
    val = np.array([t in held for t in tiles])
    a.output.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import torchvision
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)
    counts = np.bincount(labels[~val], minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor(np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** .5, 0.), dtype=torch.float32, device=device)
    weights = weights / weights[weights > 0].mean()
    train_loader = DataLoader(Crops(images[~val], labels[~val], True), batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(Crops(images[val], labels[val], False), batch_size=256, shuffle=False, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=a.lr, total_steps=a.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)
    history, best = [], None
    started = time.time()
    for epoch in range(a.epochs):
        model.train(); total = 0.; n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step(); schedule.step()
            total += float(loss) * len(y); n += len(y)
        model.eval(); correct = 0; m = 0; confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                for t, q in zip(y.numpy(), pred):
                    confusion[t, q] += 1
                correct += int((pred == y.numpy()).sum()); m += len(y)
        per_class = {CLASSES[i]: (int(confusion[i, i]), int(confusion[i].sum())) for i in range(len(CLASSES)) if confusion[i].sum()}
        row = dict(epoch=epoch + 1, train_loss=total / max(n, 1), val_accuracy=correct / max(m, 1), per_class=per_class, seconds=time.time() - started)
        history.append(row); print(json.dumps(row), flush=True)
        if best is None or row['val_accuracy'] >= best:
            best = row['val_accuracy']
            torch.save(dict(state_dict=model.state_dict(), classes=CLASSES, size=int(images.shape[1]), epoch=epoch + 1), a.output / 'best.pt')
    torch.save(dict(state_dict=model.state_dict(), classes=CLASSES, size=int(images.shape[1]), epoch=a.epochs), a.output / 'last.pt')
    (a.output / 'history.json').write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()}, classes=CLASSES, train=int((~val).sum()), val=int(val.sum()),
                                                            held_tiles=len(held), history=history, best_val_accuracy=best), indent=1))


if __name__ == '__main__':
    main()
