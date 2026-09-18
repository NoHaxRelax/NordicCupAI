"""Freeze reviewed regions into portable detector tiles and classifier crops.

Preparation is allowed before release. Training separately checks release.json.
Use the corrected export when configured; raw manual tracks may be rejected later.
Uses integer-factor BOX reduction, equal to area averaging for these factors.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil

from common import covered, fingerprint, frame_split, intersect, read, sha, write


def load_sources(root, config):
    paths = sorted((root/'data/drone/reference/helsinki/annotations').glob('*.json'))
    paths += sorted((root/config.get('validation_annotation_dir','data/drone/training/manual-validation')).glob('*.json'))
    ledger_path = root/'artifacts/drone-validation-coverage/coverage-ledger.json'
    if not paths or not ledger_path.exists():
        raise ValueError('Reference/manual annotations and region-review ledger are required')
    hashes = {str(p.relative_to(root)): sha(p) for p in paths + [ledger_path]}
    frames = {}
    track_splits = defaultdict(set)
    for path in paths:
        payload = read(path)
        reference = '/reference/' in path.as_posix()
        for a in payload.get('annotations', []):
            frame = payload['frame'] if reference else a['frame']
            cls = a['object_id'] if reference else a['class']
            if cls not in config['classes']:
                raise ValueError('Unknown class: ' + cls)
            bbox=a['bbox'] if reference else a['bbox_source_xyxy']
            if len(bbox)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in bbox) or bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:
                raise ValueError('Invalid annotation box in '+str(path))
            track = 'reference:' + cls if reference else config['track_aliases'].get(path.stem, path.stem)
            split = 'train' if reference else frame_split(frame, config)
            reviewed = reference or a.get('review_status', '').startswith('direct')
            # Projected rows never supervise a model and already exclude any
            # intersecting detector tile. Optionally avoid treating those
            # excluded rows as an evaluated occurrence of a physical track.
            if split and (reviewed or not config.get('split_reviewed_occurrences_only', False)):
                track_splits[track].add(split)
            key = ('reference' if reference else 'validation', frame)
            item = frames.setdefault(key, dict(source=key[0], frame=frame, split=split, annotations=[]))
            item['annotations'].append(dict(class_name=cls, track=track,
                bbox=bbox,
                reviewed=reviewed,
                provenance='organizer' if reference else a.get('review_status', 'unknown')))
    crossing = {t for t, splits in track_splits.items() if len(splits) > 1}
    ledger = read(ledger_path)
    if ledger.get('completion', {}).get('safe_for_reviewed_region_negative_training') is not True:
        raise ValueError('Reviewed-region completeness gate has not passed')
    regions = defaultdict(list)
    for sheet in ledger['sheets']:
        if all(sheet.get(k, {}).get('status') in ['reviewed_empty', 'reviewed_objects_resolved']
               for k in ['primary_review', 'small_object_review']):
            regions[sheet['frame']].extend(sheet['source_regions_xyxy'])
            # Empty reviewed frames are essential negatives, even without a JSON annotation file.
            frames.setdefault(('validation', sheet['frame']), dict(source='validation', frame=sheet['frame'],
                              split=frame_split(sheet['frame'], config), annotations=[]))
    return frames, regions, crossing, hashes


def windows(region, zoom):
    width = (3840, 1920, 960)[zoom]
    height = min(region[3]-region[1], (2160, 1080, 540)[zoom])
    width = min(width, region[2]-region[0])
    def starts(low, high, size):
        values = list(range(low, high-size+1, max(1, size*3//4)))
        return sorted(set(values + [high-size]))
    for y in starts(region[1], region[3], height):
        for x in starts(region[0], region[2], width):
            yield [x, y, x+width, y+height]


def build(args):
    from PIL import Image
    config = read(args.config)
    frames, regions, crossing, hashes = load_sources(args.root, config)
    identity = fingerprint(dict(sources=hashes, config=config))
    args.output.mkdir(parents=True, exist_ok=False)
    for rel in hashes:
        dest = args.output/'sources'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.root/rel, dest)
    write(args.output/'config.json', config)
    records, skipped, source_images = [], Counter(), {}

    def save_image(image, rel, row):
        path = args.output/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        row.update(file=rel, sha256=sha(path), dimensions=list(image.size))
        records.append(row)

    for (source, frame), info in sorted(frames.items()):
        split = info['split']
        if not split:
            continue
        image_path = args.root / (f'data/drone/reference/helsinki/images/frame_{frame:06d}.png' if source=='reference'
                                 else f'data/drone/reconstructed-validation/frame_{frame:06d}.png')
        if not image_path.exists():
            raise ValueError('Missing source image: ' + str(image_path))
        image = Image.open(image_path)
        if image.size != (3840, 2160):
            raise ValueError('Unexpected source dimensions')
        if image.mode == 'RGBA' and image.getchannel('A').getextrema()[0] != 255:
            skipped['incomplete_source_frame'] += 1
            continue
        image = image.convert('RGB')
        source_images[str(image_path.relative_to(args.root))] = sha(image_path)
        anns = info['annotations']
        # Deduplicate identical rows; never remove different objects merely because classes match.
        unique = {(a['track'], tuple(a['bbox'])):a for a in anns}
        anns = list(unique.values())
        available = [[0,0,3840,2160]] if source=='reference' else regions.get(frame, [])
        if covered([0,0,3840,2160], available):
            scope = [[0,0,3840,2160]]
        elif covered([0,0,3840,540], available):
            scope = [[0,0,3840,540]]
        else:
            scope = available
        for z in range(3):
            factor = (4,2,1)[z]
            seen_windows = set()
            for region in scope:
                for box in windows(region,z):
                    if tuple(box) in seen_windows:
                        continue
                    seen_windows.add(tuple(box))
                    visible = [(a,intersect(a['bbox'],box)) for a in anns]
                    visible = [(a,b) for a,b in visible if b]
                    if any(not a['reviewed'] or a['track'] in crossing for a,b in visible):
                        skipped['tile_contains_unreviewed_or_cross_split_track'] += 1
                        continue
                    w,h=box[2]-box[0],box[3]-box[1]
                    tile=image.crop(box).resize((w//factor,h//factor),Image.Resampling.BOX)
                    stem=f'{source}-{frame:06d}-L{z}-{box[0]}-{box[1]}'
                    label=f'detector/labels/{split}/{stem}.txt'
                    rows=[]
                    for a,b in visible:
                        rows.append(f"{config['classes'].index(a['class_name'])} {(b[0]+b[2]-2*box[0])/(2*w):.9f} {(b[1]+b[3]-2*box[1])/(2*h):.9f} {(b[2]-b[0])/w:.9f} {(b[3]-b[1])/h:.9f}")
                    path=args.output/label;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text('\n'.join(rows)+('\n' if rows else ''))
                    save_image(tile,f'detector/images/{split}/{stem}.png',dict(task='detector',split=split,source=source,
                        frame=frame,zoom=z,source_region=box,label_file=label,label_sha256=sha(path),
                        classes=[a['class_name'] for a,b in visible],tracks=[a['track'] for a,b in visible]))
                    # Verified negatives for the crop recogniser. Match positive crop sizes instead of
                    # teaching that background means a large panoramic input.
                    if not visible:
                        size=min(96,tile.width,tile.height)
                        x=(tile.width-size)//2;y=(tile.height-size)//2
                        save_image(tile.crop((x,y,x+size,y+size)),f'classifier/{split}/background/{stem}.png',
                            dict(task='classifier',split=split,source=source,frame=frame,zoom=z,
                                 class_name='background',track=None))
            if frame % config['classifier_stride'] != 0 and source != 'reference':
                continue
            # Classifier positives do not claim that the source frame is exhaustively labelled.
            small=image.resize((3840//factor,2160//factor),Image.Resampling.BOX)
            for index,a in enumerate(anns):
                if not a['reviewed'] or a['track'] in crossing:
                    continue
                b=intersect(a['bbox'],[0,0,3840,2160])
                if not b:
                    continue
                margin=max(2,0.12*max(b[2]-b[0],b[3]-b[1]))
                crop=[max(0,int((b[0]-margin)//factor)),max(0,int((b[1]-margin)//factor)),
                      min(small.width,int((b[2]+margin+factor-1)//factor)),min(small.height,int((b[3]+margin+factor-1)//factor))]
                stem=f'{source}-{frame:06d}-L{z}-{index}'
                save_image(small.crop(crop),f"classifier/{split}/{a['class_name']}/{stem}.png",
                           dict(task='classifier',split=split,source=source,frame=frame,zoom=z,class_name=a['class_name'],track=a['track']))
    counts=Counter(r['task']+'/'+r['split'] for r in records)
    coverage={}
    for task in ['detector','classifier']:
        for split in ['train','dev']:
            selected=[r for r in records if r['task']==task and r['split']==split]
            byclass=Counter(c for r in selected for c in (r['classes'] if task=='detector' else [r['class_name']]))
            coverage[task+'/'+split]=dict(appearances=dict(byclass),missing_classes=[c for c in config['classes'] if not byclass[c]],
                unique_tracks=len({t for r in selected for t in (r['tracks'] if task=='detector' else [r['track']]) if t}))
    for task in ['detector','classifier']:
        if not counts[task+'/train'] or not counts[task+'/dev']:
            raise ValueError('Empty train/dev partition: '+task)
    # Identical backgrounds must not occur on both sides of the partition.
    hashes_to_splits=defaultdict(set)
    for r in records:
        hashes_to_splits[(r['task'],r['sha256'])].add(r['split'])
    duplicate_hashes={k for k,v in hashes_to_splits.items() if len(v)>1}
    if duplicate_hashes:
        raise ValueError(f'{len(duplicate_hashes)} exact image duplicates cross the split; revise partitions')
    if any(sha(args.root/rel)!=digest for rel,digest in hashes.items()):
        raise ValueError('Source annotations changed during preparation. Build a new snapshot after the handoff.')
    manifest=dict(schema=1,source_fingerprint=identity,source_hashes=hashes,source_image_hashes=source_images,
        classes=config['classes'],classifier_classes=config['classes']+['background'],config=config,
        counts=dict(counts),coverage=coverage,excluded_cross_split_tracks=sorted(crossing),skipped=dict(skipped),records=records,
        limitations=['Participant labels and review ledger, not organizer validation ground truth.',
        f"Dev is a later geographic/temporal portion of the same flight, separated by a {config['validation_dev_frames'][0]-config['validation_train_frames'][1]-1}-frame buffer.",
        'Missing dev classes cannot be evaluated; do not call subset AP a 16-class benchmark.',
        'Only explicitly reviewed tiles supply detector background. Algorithmic lower-frame completions are excluded.',
        'Classifier positives may come from incompletely labelled frames; their crop metric is not discovery recall.',
        'The dev partition is for selection and changes, not an untouched final test. No final evaluation calls.'])
    write(args.output/'manifest.json',manifest)
    write(args.output/'summary.json',{k:v for k,v in manifest.items() if k not in ['records','source_hashes','source_image_hashes']})
    print(json.dumps(dict(output=str(args.output),source_fingerprint=identity,counts=counts,coverage=coverage)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    build(p.parse_args())
