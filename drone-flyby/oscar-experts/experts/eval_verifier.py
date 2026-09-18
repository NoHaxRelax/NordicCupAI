"""Evaluate a trained verifier on a crop set it never trained on (e.g. synthetic-validation crops).

Reports overall accuracy, per-class recall, background rejection and the confusion matrix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train_verifier import CLASSES, Crops, load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--crops', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    import torchvision
    checkpoint = torch.load(a.weights, map_location='cpu', weights_only=False)
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['classes']))
    model.load_state_dict(checkpoint['state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    images, labels, _ = zip(*[load(c) for c in a.crops])
    images, labels = np.concatenate(images), np.concatenate(labels)
    loader = torch.utils.data.DataLoader(Crops(images, labels, False), batch_size=256, shuffle=False)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    probs = []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).softmax(1).cpu().numpy()
            probs.append(out)
            for t, q in zip(y.numpy(), out.argmax(1)):
                confusion[t, q] += 1
    probs = np.concatenate(probs)
    per_class = {CLASSES[i]: dict(recall=round(confusion[i, i] / confusion[i].sum(), 3), n=int(confusion[i].sum()),
                                  called_background=int(confusion[i, 0])) for i in range(len(CLASSES)) if confusion[i].sum()}
    bg = confusion[0]
    report = dict(weights=str(a.weights), crops=[str(c) for c in a.crops], n=int(confusion.sum()), accuracy=round(np.trace(confusion) / confusion.sum(), 4),
                  background_rejection=round(bg[0] / bg.sum(), 4) if bg.sum() else None, per_class=per_class,
                  confusion={CLASSES[i]: {CLASSES[j]: int(confusion[i, j]) for j in range(len(CLASSES)) if confusion[i, j]} for i in range(len(CLASSES)) if confusion[i].sum()})
    print(json.dumps(dict(accuracy=report['accuracy'], background_rejection=report['background_rejection'], per_class=per_class), indent=1))
    if a.output:
        a.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
