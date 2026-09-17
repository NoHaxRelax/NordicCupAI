"""Render full-context camera views around frozen manual holdout targets."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from prepare import digest, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--training-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'images').mkdir()
    manifest_path = args.training_data/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    targets = [r for r in manifest['records'] if r['task']=='classifier'
               and r['split']=='val' and r['zoom']==1]
    snapshots = Path(__file__).parent/'snapshots'
    annotations, sources, images, records = {}, {}, {}, []
    rng = np.random.default_rng(170927)
    for target in targets:
        group, frame = target['group'], target['frame']
        if group not in annotations:
            path = snapshots/'manual'/f'{group}.json'
            assert digest(path) == manifest['annotation_snapshots'][f'manual/{group}.json']
            annotations[group] = json.loads(path.read_text())['annotations']
        ann = next(a for a in annotations[group] if a['frame']==frame and a['seed_id']==target['seed_id'])
        relative = f'data/drone/reconstructed-validation/frame_{frame:06d}.png'
        if frame not in images:
            path = args.source_root/relative
            source_hash = digest(path)
            assert source_hash == manifest['inputs'][relative]
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert image is not None and image.shape == (2160,3840,4)
            assert np.all(image[:,:,3] == 255), 'Partial native reconstruction'
            images[frame] = image[:,:,:3]
            sources[relative] = source_hash
        for zoom in range(3):
            view, region, scale = render(images[frame], ann['bbox_source_xyxy'], zoom,
                                        jitter=rng.uniform(-.15,.15,2))
            box = [(ann['bbox_source_xyxy'][i]-region[i%2])*scale for i in range(4)]
            assert 0 <= box[0] < box[2] <= 960 and 0 <= box[1] < box[3] <= 540
            file = f'images/{group}_{frame:06d}_L{zoom}.png'
            assert cv2.imwrite(str(args.output/file), view)
            records.append(dict(id=Path(file).stem, subset='manual_known_positives', file=file,
                                sha256=digest(args.output/file), frame=frame, group=group, zoom=zoom,
                                source_region=region, boxes=[dict(class_name=target['class_name'], xyxy=box)],
                                target_seed_id=target['seed_id'], source_box=ann['bbox_source_xyxy']))
    output = dict(classes=manifest['classes'], training_manifest_sha256=digest(manifest_path),
                  seed=170927, inputs=sources, records=records,
                  limits=['Positive-only manual development annotations, not organizer ground truth.',
                          'Targets are the frozen classifier holdout; detector trained only on organizer reference views.',
                          'Camera windows are selected using target boxes. This is visible-target recognition, not autonomous discovery.',
                          '71 appearances from six physical tracks, correlated across time and zoom; not 213 independent objects.',
                          'All unannotated detections are unknown, not false positives. No precision or mAP for this subset.'])
    (args.output/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(views=len(records), tracks=dict(Counter(r['group'] for r in records)),
                          png_megabytes=sum(p.stat().st_size for p in (args.output/'images').glob('*.png'))/1e6)))


if __name__ == '__main__':
    main()
