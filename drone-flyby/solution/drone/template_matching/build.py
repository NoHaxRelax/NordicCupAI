"""Build a frozen bank from reference images and optional bounded validation tracks."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import sha

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_HOLDOUT = (5, 12, 20)


def build(root, output, views=5, validation_through=None, validation_views=3, foreground_mask=False):
    if views < 1:
        raise ValueError('views must be positive')
    reference = root/'data/drone/reference/helsinki'
    grouped = defaultdict(list)
    source_hashes = {}
    for path in sorted((reference/'annotations').glob('*.json')):
        document = json.loads(path.read_text())
        source_hashes[str(path.relative_to(root))] = sha(path)
        if document['frame'] in REFERENCE_HOLDOUT:
            continue
        for a in document['annotations']:
            x1, y1, x2, y2 = a['bbox']
            if 0 < x1 < x2 < 3839 and 0 < y1 < y2 < 2159:
                grouped[a['object_id']].append((document['frame'], a['bbox']))
    if len(grouped) != 16:
        raise ValueError(f'Expected all 16 reference classes, found {len(grouped)}')
    if validation_through is not None and not 5 <= validation_through <= 149:
        raise ValueError('Validation bank cutoff must be 5..149; late holdout is reserved')
    if validation_views < 1:
        raise ValueError('validation_views must be positive')
    output.mkdir(parents=True, exist_ok=False)
    records, cache = [], {}
    for label, candidates in sorted(grouped.items()):
        # Spread views across the flight, without peeking at validation performance.
        for index in sorted(set(np.linspace(0, len(candidates)-1, min(views, len(candidates))).round().astype(int))):
            frame, bbox = candidates[index]
            path = reference/'images'/f'frame_{frame:06d}.png'
            if frame not in cache:
                cache[frame] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                source_hashes[str(path.relative_to(root))] = sha(path)
            image = cache[frame]
            if image is None or image.shape[:2] != (2160, 3840):
                raise ValueError('Missing or invalid source image: '+str(path))
            x1, y1 = np.floor(bbox[:2]).astype(int)
            x2, y2 = np.ceil(bbox[2:]).astype(int)
            patch = image[y1:y2, x1:x2]
            if patch.shape[2] == 4 and np.any(patch[:,:,3] != 255):
                raise ValueError('Transparent training template')
            name = f'{label}-{frame:06d}'
            dest = output/(name+'.png')
            if not cv2.imwrite(str(dest), patch[:,:,:3]):
                raise OSError(str(dest))
            records.append(dict(id=name, file=dest.name, sha256=sha(dest),
                                **{'class': label}, frame=frame, bbox=list(map(int, (x1,y1,x2,y2))),
                                source='organizer_reference', group='helsinki:'+label))
    validation_tracks = []
    if validation_through is not None:
        # Group linked fragments before partitioning, including projected tails when
        # determining physical track extent. Only directly reviewed crops enter bank.
        aliases = json.loads((root/'drone/overnight/config.json').read_text())['track_aliases']
        source_hashes['drone/overnight/config.json'] = sha(root/'drone/overnight/config.json')
        tracks = defaultdict(list)
        paths = {}
        for path in sorted((root/'data/drone/training/algorithmic-full-validation').glob('*.json')):
            doc = json.loads(path.read_text())
            if not doc.get('annotations'):
                continue
            track = aliases.get(doc.get('track_id',path.stem), doc.get('track_id',path.stem))
            tracks[track].extend(doc['annotations'])
            paths.setdefault(track, []).append(path)
        selected = defaultdict(list)
        for track, annotations in sorted(tracks.items()):
            if max(a['frame'] for a in annotations) > validation_through:
                continue
            for a in annotations:
                b = a['bbox_source_xyxy']
                if a.get('provenance') == 'reviewed_positive' and a.get('review_status','').startswith('direct') and 0 < b[0] < b[2] < 3839 and 0 < b[1] < b[3] < 2159:
                    selected[a['class']].append((track, a))
        for label, candidates in sorted(selected.items()):
            candidates.sort(key=lambda pair: (pair[0], pair[1]['frame']))
            indices = sorted(set(np.linspace(0,len(candidates)-1,min(validation_views,len(candidates))).round().astype(int)))
            for index in indices:
                track, a = candidates[index]
                frame = a['frame']
                path = root/f'data/drone/reconstructed-validation/frame_{frame:06d}.png'
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape[:2] != (2160,3840):
                    raise ValueError('Missing validation source '+str(path))
                x1,y1 = np.floor(a['bbox_source_xyxy'][:2]).astype(int)
                x2,y2 = np.ceil(a['bbox_source_xyxy'][2:]).astype(int)
                patch = image[y1:y2,x1:x2]
                if patch.shape[2] == 4 and np.any(patch[:,:,3] != 255):
                    raise ValueError('Unobserved validation template pixels')
                name = f'validation-{label}-{frame:06d}-{index}'
                dest = output/(name+'.png')
                if not cv2.imwrite(str(dest), patch[:,:,:3]):
                    raise OSError(str(dest))
                source_hashes[str(path.relative_to(root))] = sha(path)
                for source in paths[track]:
                    source_hashes[str(source.relative_to(root))] = sha(source)
                validation_tracks.append(track)
                records.append(dict(id=name,file=dest.name,sha256=sha(dest), **{'class':label},
                                    frame=frame,bbox=list(map(int,(x1,y1,x2,y2))),
                                    source='reviewed_validation',group=track))
    if foreground_mask:
        for record in records:
            image = cv2.imread(str(output/record['file']))
            h,w = image.shape[:2]
            labels = np.zeros((h,w),np.uint8)
            cv2.setRNGSeed(0)
            cv2.grabCut(image,labels,(1,1,w-2,h-2),np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_RECT)
            mask = np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
            fraction = float(np.mean(mask>0))
            # Include silhouette boundaries, where high-pass signal is strongest.
            if fraction < .04 or fraction > .95:
                mask[:] = 255
                record['mask_fallback'] = 'Unreliable foreground fraction'
            else:
                mask = cv2.dilate(mask,np.ones((5,5),np.uint8))
            path = output/(record['id']+'-mask.png')
            if not cv2.imwrite(str(path),mask):
                raise OSError(str(path))
            record.update(mask_file=path.name,mask_sha256=sha(path),foreground_fraction=fraction)
    manifest = dict(schema=1, templates=records, source_hashes=source_hashes, foreground_mask=foreground_mask,
                    reference_holdout_frames=list(REFERENCE_HOLDOUT),
                    validation_templates=sum(r['source']=='reviewed_validation' for r in records),
                    validation_train_through=validation_through,
                    validation_train_tracks=sorted(set(validation_tracks)),
                    validation_policy='Whole tracks ending by cutoff only; late frames 181-249 reserved. No projected crops.',
                    classes=dict(Counter(r['class'] for r in records)),
                    versions={'opencv': cv2.__version__, 'numpy': np.__version__})
    if any(sha(root/path) != digest for path,digest in source_hashes.items()):
        raise ValueError('Source changed during preparation; rebuild into a new directory')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--views', type=int, default=5)
    p.add_argument('--validation-through', type=int, help='Optional early-frame cutoff, maximum 149')
    p.add_argument('--validation-views', type=int, default=3, help='Maximum validation templates per class')
    p.add_argument('--foreground-mask', action='store_true', help='Deterministic foreground segmentation to reduce background matching')
    a = p.parse_args()
    m = build(a.root, a.output, a.views, a.validation_through, a.validation_views, a.foreground_mask)
    print(json.dumps({'templates': len(m['templates']), 'classes': m['classes'], 'validation_templates': m['validation_templates']}))


if __name__ == '__main__':
    main()
