"""Freeze an auditable first-run dataset without treating missing labels as negatives."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = sorted('condor hangar helicopter jammer jet_plane large_launcher large_tower medium_launcher medium_plane mine_roller small_launcher small_plane small_tower spacecraft ta-ta tank'.split())
MANUAL_TRAIN = {'annotations', 'hangar-053-082', 'jet-plane-c-040-082'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(image, box, zoom, jitter=(0, 0)):
    h, w = image.shape[:2]
    rw, rh = (3840 // (2**zoom), 2160 // (2**zoom))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x = max(0, min(w - rw, round(cx - rw / 2 + jitter[0] * rw)))
    y = max(0, min(h - rh, round(cy - rh / 2 + jitter[1] * rh)))
    view = cv2.resize(image[y:y+rh, x:x+rw], (960, 540), interpolation=cv2.INTER_AREA)
    return view, (x, y, x+rw, y+rh), 960 / rw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    snapshot = Path(__file__).parent / 'snapshots'
    rng = np.random.default_rng(170926)
    records, inputs = [], {}

    def save(image, relative, metadata, label=None):
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert image.size and cv2.imwrite(str(dest), image)
        record = dict(metadata, file=relative, sha256=digest(dest), dimensions=[image.shape[1], image.shape[0]])
        if label is not None:
            lp = out / relative.replace('/images/', '/labels/').replace('.png', '.txt')
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(label)
            record.update(label_file=str(lp.relative_to(out)), label_sha256=digest(lp))
        records.append(record)

    def crop(image, box, zoom):
        # Reject partial or reconstructed-transparent objects and context.
        h, w = image.shape[:2]
        if box[0] <= 0 or box[1] <= 0 or box[2] >= w or box[3] >= h:
            return None
        view, region, scale = render(image, box, zoom)
        vb = [(box[0]-region[0])*scale, (box[1]-region[1])*scale,
              (box[2]-region[0])*scale, (box[3]-region[1])*scale]
        a, b = max(0, math.floor(vb[0])-8), max(0, math.floor(vb[1])-8)
        c, d = min(960, math.ceil(vb[2])+8), min(540, math.ceil(vb[3])+8)
        patch = view[b:d, a:c]
        if patch.shape[2] == 4:
            if np.any(patch[:, :, 3] != 255):
                return None
            patch = patch[:, :, :3]
        return patch

    for path in sorted((snapshot / 'reference').glob('*.json')):
        doc = json.loads(path.read_text())
        src = args.source_root / 'data/drone/reference/helsinki/images' / path.with_suffix('.png').name
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        assert image is not None and image.shape[:2] == (2160, 3840)
        image = image[:, :, :3]
        inputs[str(src.relative_to(args.source_root))] = digest(src)
        anns, frame = doc['annotations'], doc['frame']
        for k, zoom in enumerate([0] + [1]*7 + [2]*2):
            anchor = anns[k % len(anns)]['bbox']
            view, region, scale = render(image, anchor, zoom, rng.uniform(-.3, .3, 2))
            labels = []
            for a in anns:
                x1, y1, x2, y2 = a['bbox']
                x1, x2 = np.clip([(x1-region[0])*scale, (x2-region[0])*scale], 0, 960)
                y1, y2 = np.clip([(y1-region[1])*scale, (y2-region[1])*scale], 0, 540)
                if x2 > x1 and y2 > y1:
                    labels.append(f'{CLASSES.index(a["object_id"])} {(x1+x2)/1920:.8f} {(y1+y2)/1080:.8f} {(x2-x1)/960:.8f} {(y2-y1)/540:.8f}')
            save(view, f'detector/images/train/ref_{frame:06d}_{k:02d}_L{zoom}.png',
                 dict(task='detector', split='train', frame=frame, zoom=zoom, source='organizer_reference', source_region=region),
                 '\n'.join(labels)+'\n' if labels else '')
        for i, ann in enumerate(anns):
            for zoom in range(3):
                patch = crop(image, ann['bbox'], zoom)
                if patch is not None:
                    cls = ann['object_id']
                    save(patch, f'classifier/train/{cls}/ref_{frame:06d}_{i}_L{zoom}.png',
                         dict(task='classifier', split='train', frame=frame, zoom=zoom, group=f'helsinki-{cls}', class_name=cls, source='organizer_reference'))

    skipped = Counter()
    for path in sorted((snapshot / 'manual').glob('*.json')):
        doc = json.loads(path.read_text())
        if path.stem.startswith('medium-plane-'):
            skipped['manual_medium_plane_failed_visual_QA'] += len(doc['annotations'])
            continue
        split = 'train' if path.stem in MANUAL_TRAIN else 'val'
        for i, ann in enumerate(doc['annotations']):
            if i % 3:
                continue  # Thin correlated adjacent frames; never split them at random.
            assert ann['evidence'] == 'manual_reference_match'
            assert ann['review_status'].startswith('directly_reviewed')
            frame, cls = ann['frame'], ann['class']
            assert frame >= 5 and cls in CLASSES
            src = args.source_root / 'data/drone/reconstructed-validation' / f'frame_{frame:06d}.png'
            image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            assert image is not None and image.shape[:2] == (2160, 3840)
            inputs[str(src.relative_to(args.source_root))] = digest(src)
            for zoom in range(3):
                patch = crop(image, ann['bbox_source_xyxy'], zoom)
                if patch is None:
                    skipped['edge_or_transparency'] += 1
                    continue
                save(patch, f'classifier/{split}/{cls}/{path.stem}_{frame:06d}_L{zoom}.png',
                     dict(task='classifier', split=split, frame=frame, zoom=zoom, group=path.stem,
                          class_name=cls, source='manual_positive', seed_id=ann['seed_id']))

    groups = {s: {r['group'] for r in records if r['task']=='classifier' and r['split']==s} for s in ('train', 'val')}
    assert not groups['train'] & groups['val']
    assert set(r['class_name'] for r in records if r['task']=='classifier' and r['split']=='train') == set(CLASSES)
    assert Counter(r['zoom'] for r in records if r['task']=='detector') == {0:25, 1:175, 2:50}
    # Ultralytics requires a val path. It is explicitly a training diagnostic,
    # not independent validation; checkpoint selection uses the last epoch.
    (out/'detector/data.yaml').write_text('path: .\ntrain: images/train\nval: images/train\nnames: '+json.dumps(CLASSES)+'\n')
    manifest = dict(schema_version=1, classes=CLASSES, seed=170926, inputs=inputs,
                    annotation_snapshots={str(p.relative_to(snapshot)):digest(p) for p in snapshot.rglob('*.json')},
                    skipped=dict(skipped), counts=dict(Counter(f'{r["task"]}/{r["split"]}' for r in records)),
                    limits=['Detector has no independent fully labeled holdout; reported detector validation is training-fit only.',
                            'Classifier holdout contains whole manual tracks, but shares the validation scene with manual training tracks.',
                            'Manual labels are positive pseudo-labels and incomplete; no manual full-frame detector training.',
                            'Classifier accuracy does not measure object discovery or competition mAP.',
                            'Helicopter-b tracks excluded because their annotation correction was still active at snapshot time.',
                            'All manual medium-plane tracks excluded: native-resolution sampled crops failed visual QA against the reference sprite.'],
                    records=records)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('records','inputs','annotation_snapshots')}, indent=2))


if __name__ == '__main__':
    main()
