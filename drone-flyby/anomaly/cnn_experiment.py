"""Train a tiny class-free CNN to rerank anomaly proposals."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

try:
    from .experiment import VIEW_H, VIEW_W, average_precision, iou, load_frames, render
except ImportError:  # Direct execution: python anomaly/cnn_experiment.py
    from experiment import VIEW_H, VIEW_W, average_precision, iou, load_frames, render


def crop64(image, box):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.shape[1], int(np.ceil(x2))), min(image.shape[0], int(np.ceil(y2)))
    patch = image[y1:y2, x1:x2]
    if min(patch.shape[:2]) < 2:
        return np.full((64, 64, 3), 127, np.uint8)
    h, w = patch.shape[:2]
    scale = 56 / max(h, w)
    patch = cv2.resize(patch, (max(1, round(w*scale)), max(1, round(h*scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((64, 64, 3), 127, np.uint8)
    h, w = patch.shape[:2]
    canvas[(64-h)//2:(64-h)//2+h, (64-w)//2:(64-w)//2+w] = patch
    return canvas


class ObjectnessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for out in (16, 32, 64, 96):
            layers += [nn.Conv2d(channels, out, 3, 2, 1), nn.BatchNorm2d(out), nn.SiLU()]
            channels = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1))

    def forward(self, value):
        return self.head(self.features(value)).squeeze(1)


def make_training(scene, train_ids, seed=180926):
    rng = np.random.default_rng(seed)
    positives, negatives = [], []
    for _, annotations, image in load_frames(scene):
        for annotation in annotations:
            if annotation['object_id'] not in train_ids:
                continue
            for zoom in (0, 1, 2):
                view, target, _ = render(image, annotation['bbox'], zoom)
                cx, cy = (target[0]+target[2])/2, (target[1]+target[3])/2
                w, h = target[2]-target[0], target[3]-target[1]
                for extent in (1.0, 1.5, 2.2):
                    jx, jy = rng.normal(0, .08, 2)
                    box = [cx-w*extent*(.5+jx), cy-h*extent*(.5+jy),
                           cx+w*extent*(.5-jx), cy+h*extent*(.5-jy)]
                    positives.append(crop64(view, box))
                for _ in range(5):
                    for attempt in range(100):
                        nw = float(np.clip(w*rng.uniform(.7, 3.0), 7, 300))
                        nh = float(np.clip(h*rng.uniform(.7, 3.0), 7, 300))
                        x = float(rng.uniform(0, VIEW_W-nw)); y = float(rng.uniform(0, VIEW_H-nh))
                        box = [x, y, x+nw, y+nh]
                        if iou(box, target) < .01:
                            negatives.append(crop64(view, box)); break
    return np.stack(positives), np.stack(negatives)


def train_model(positives, negatives, steps, seed=180926):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    model = ObjectnessCNN(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    for step in range(steps):
        pi = rng.integers(0, len(positives), 32); ni = rng.integers(0, len(negatives), 32)
        batch = np.concatenate((positives[pi], negatives[ni])).astype(np.float32)
        labels = np.r_[np.ones(32), np.zeros(32)].astype(np.float32)
        order = rng.permutation(64); batch, labels = batch[order], labels[order]
        if rng.random() < .5: batch = batch[:, :, ::-1].copy()
        batch = np.clip(batch*rng.uniform(.85, 1.15)+rng.uniform(-10, 10), 0, 255)
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2)/255).float(); y = torch.from_numpy(labels)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward(); optimizer.step()
        if step % 100 == 0 or step+1 == steps:
            print(json.dumps(dict(step=step+1, loss=float(loss.detach()),
                                  batch_accuracy=float(((logits > 0) == (y > .5)).float().mean()))), flush=True)
    return model.eval()


def score_views(model, report, scene):
    frames = {frame: (annotations, image) for frame, annotations, image in load_frames(scene)}
    output = []
    with torch.inference_mode():
        for row in report['view_results']:
            annotations, source = frames[row['frame']]
            left, top, right, bottom = row['view_region']
            view = source[top:bottom, left:right]
            if view.shape[:2] != (VIEW_H, VIEW_W):
                view = cv2.resize(view, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
            predictions = row['predictions']
            crops = np.stack([crop64(view, item[1]) for item in predictions])
            scores = []
            for first in range(0, len(crops), 128):
                x = torch.from_numpy(crops[first:first+128].transpose(0, 3, 1, 2).copy()/255).float()
                scores.extend(model(x).sigmoid().tolist())
            rescored = sorted([[float(score), item[1]] for score, item in zip(scores, predictions)], reverse=True)
            output.append({**row, 'predictions': rescored,
                           'anchor_object_id': None if row['anchor'] is None else annotations[row['anchor']]['object_id']})
    return output


def heldout_ap(views, heldout, threshold, budget):
    prepared = []
    for view in views:
        if view['anchor_object_id'] is not None and view['anchor_object_id'] not in heldout:
            continue
        # Known training objects are ignored, not mislabeled as background.
        truth = [target for target in view['truth'] if target['object_id'] in heldout]
        ignored = [target for target in view['truth'] if target['object_id'] not in heldout]
        predictions = [(score, box) for score, box in view['predictions']
                       if not any(iou(box, target['box']) >= threshold for target in ignored)]
        prepared.append({**view, 'truth': truth, 'predictions': predictions})
    return average_precision(prepared, threshold, budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, default=Path(__file__).parents[1]/'src/helsinki')
    parser.add_argument('--proposals', type=Path, default=Path(__file__).parent/'proposals.json.gz')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'cnn-results.json')
    parser.add_argument('--weights', type=Path, default=Path(__file__).parent/'objectness-cnn.pt')
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    with gzip.open(args.proposals, 'rt') as stream:
        report = json.load(stream)
    object_ids = sorted({target['object_id'] for view in report['view_results'] for target in view['truth']})
    train_ids, heldout = object_ids[::2], object_ids[1::2]
    positives, negatives = make_training(args.scene, set(train_ids))
    model = train_model(positives, negatives, args.steps)
    torch.save(dict(state_dict=model.state_dict(), train_object_ids=train_ids,
                    heldout_object_ids=heldout, steps=args.steps), args.weights)
    views = score_views(model, report, args.scene)
    metrics = {}
    for zoom in range(3):
        selected = [view for view in views if view['zoom'] == zoom]
        for budget in (25, 100, 300):
            for threshold in (.25, .50):
                key = f'L{zoom}/budget{budget}/IoU{threshold:.2f}'
                metrics[key] = dict(all_objects=average_precision(selected, threshold, budget),
                                    heldout_objects=heldout_ap(selected, set(heldout), threshold, budget))
    result = dict(method='binary object-vs-background CNN reranking anomaly proposals',
                  split='alternating sorted object identities; all appearances of an identity remain together',
                  train_object_ids=train_ids, heldout_object_ids=heldout,
                  training_crops=dict(positive=len(positives), negative=len(negatives)), metrics=metrics)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
