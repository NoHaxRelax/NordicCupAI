"""Box placement for full-frame scoring: extents, edge conventions, transients.

The organizer scores COCO mAP at IoU 0.50 against boxes that are integer,
looser than the visible silhouette, and clipped at pixel index 3839/2159.
This module holds the runtime rules that turn a detector box into a box in
that convention. It reads no dataset at runtime; the size prior is a small
JSON fitted once from the public reference annotations by ``--fit``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .motion import box_array, finite

EXTENT_POLICIES = ('detector', 'blend', 'prior')
DEFAULT_PRIOR = Path(__file__).with_name('size-prior.json')
SOURCE_LAST_INDEX = (3839., 2159.)


class SizePrior:
    """Per-class organizer box size as a function of vertical image position.

    Fitted on unclipped reference boxes of one physical instance per class.
    It is a prior, not a per-instance truth: other instances of a class can
    differ by their heading. Sizes are clamped inside 0.9x the smallest and
    1.1x the largest observed box of that class.
    """
    def __init__(self, table):
        if not isinstance(table, dict) or not table:
            raise ValueError('Size prior needs a non-empty class table')
        self.table = {}
        for label, row in table.items():
            entry = {}
            for key in ('w', 'h'):
                p = row[key]
                values = [float(finite(p[k], key)) for k in ('intercept', 'slope', 'minimum', 'maximum')]
                if values[2] <= 0 or values[3] < values[2]:
                    raise ValueError(f'Invalid size prior for {label}')
                entry[key] = {**p, **dict(zip(('intercept', 'slope', 'minimum', 'maximum'), values))}
            self.table[label] = entry

    @classmethod
    def load(cls, path=DEFAULT_PRIOR):
        return cls(json.loads(Path(path).read_text())['classes'])

    def __contains__(self, label):
        return label in self.table

    def size(self, label, cy):
        """Expected organizer (width, height) in source pixels for a centre at row cy."""
        row = self.table[label]
        out = []
        for key in ('w', 'h'):
            p = row[key]
            out.append(min(1.1*p['maximum'], max(.9*p['minimum'], p['intercept']+p['slope']*float(cy))))
        return np.array(out)

    @classmethod
    def fit(cls, annotation_directory, source_size=(3840, 2160)):
        """Fit from organizer annotation JSON files; returns (prior, provenance)."""
        directory = Path(annotation_directory)
        rows = {}
        digest = hashlib.sha256()
        files = sorted(directory.glob('*.json'))
        if not files:
            raise FileNotFoundError(f'No annotation files under {directory}')
        for path in files:
            data = path.read_bytes(); digest.update(data)
            for a in json.loads(data)['annotations']:
                x1, y1, x2, y2 = a['bbox']
                if x1 <= 0 or y1 <= 0 or x2 >= source_size[0]-1 or y2 >= source_size[1]-1:
                    continue  # clipped at a frame edge: the extent is unknown
                rows.setdefault(a['object_id'], []).append(((y1+y2)/2, x2-x1, y2-y1))
        table = {}
        for label, values in rows.items():
            cy = np.array([v[0] for v in values]); entry = {}
            for index, key in ((1, 'w'), (2, 'h')):
                size = np.array([v[index] for v in values], float)
                if len(values) >= 3 and np.ptp(cy) > 1:
                    slope, intercept = np.polyfit(cy, size, 1)
                else:
                    slope, intercept = 0., float(np.median(size))
                entry[key] = {'intercept': float(intercept), 'slope': float(slope),
                              'minimum': float(size.min()), 'maximum': float(size.max()),
                              'median': float(np.median(size)), 'count': int(len(size))}
            table[label] = entry
        provenance = {'annotation_directory': str(directory), 'files': len(files),
                      'annotations_sha256': digest.hexdigest(),
                      'policy': 'Unclipped organizer boxes only; one physical instance per class; '
                                'slope fitted against box centre row when at least three views exist.'}
        return cls(table), provenance


def normalize_extent(label, box, prior, policy='detector', prior_weight=None, axes=(True, True)):
    """Return the box whose size follows the extent policy, centred on the detection.

    detector: unchanged. prior: the class prior size. blend: geometric mean of
    detection and prior sizes (``prior_weight`` 0.5), which tolerates a factor
    two disagreement in either direction at IoU 0.50. Unknown classes and a
    missing prior leave the box unchanged. ``axes`` limits the change to the
    horizontal and/or vertical extent.
    """
    box = box_array(box)
    if policy not in EXTENT_POLICIES:
        raise ValueError(f'Unknown extent policy {policy!r}')
    weight = {'detector': 0., 'blend': .5, 'prior': 1.}[policy] if prior_weight is None else float(prior_weight)
    if not 0 <= weight <= 1:
        raise ValueError('prior_weight must lie in [0,1]')
    if weight == 0 or prior is None or label not in prior or not any(axes):
        return box
    centre = (box[:2]+box[2:])/2
    size = box[2:]-box[:2]
    target = prior.size(label, centre[1])
    blended = np.where(np.asarray(axes, bool), np.exp((1-weight)*np.log(size)+weight*np.log(target)), size)
    return np.r_[centre-blended/2, centre+blended/2]


def place_partial(label, box, view_region, source_size, prior, policy='detector', prior_weight=None, margin=1.):
    """Place a detection cut by the crop or source edge.

    An axis with both sides visible follows the extent policy; an axis cut by
    the crop edge is completed towards the prior; an axis cut by the source
    edge keeps the visible extent, which is what the organizer box shows too.
    """
    box = box_array(box); region = np.asarray(view_region, float)
    cut = [box[a] <= region[a]+margin or box[2+a] >= region[2+a]-margin for a in range(2)]
    placed = normalize_extent(label, box, prior, policy, prior_weight, axes=tuple(not c for c in cut))
    return complete_partial(label, placed, view_region, source_size, prior)


def entry_box(label, box, view_region, source_size, prior, margin=1.):
    """Full extent of an object entering at the top source edge, or None.

    The visible part gives the horizontal extent and the bottom edge; the
    class prior supplies the height, so the box may start above the frame.
    All other sides must be inside the crop for the extent to be trusted.
    """
    box = box_array(box); region = np.asarray(view_region, float)
    if prior is None or label not in prior or region[1] > 0 or box[1] > margin:
        return None
    if box[0] <= region[0]+margin or box[2] >= region[2]-margin or box[3] >= region[3]-margin:
        return None
    height = prior.size(label, box[3])[1]
    return np.array([box[0], box[3]-height, box[2], box[3]])


def complete_partial(label, box, view_region, source_size, prior):
    """Extend a crop-clipped box towards the class prior size on the clipped sides.

    A side touching the delivered crop edge, but not the source-frame edge,
    hides an unknown remainder; the organizer box continues past it. Sides on
    the source edge stay put because the organizer box is clipped there too.
    """
    box = box_array(box); region = np.asarray(view_region, float)
    if prior is None or label not in prior:
        return box
    target = prior.size(label, (box[1]+box[3])/2)
    result = box.copy()
    for axis in range(2):
        missing = max(0., target[axis]-(box[2+axis]-box[axis]))
        if missing <= 0:
            continue
        low_clipped = box[axis] <= region[axis]+1 and region[axis] > 0
        high_clipped = box[2+axis] >= region[2+axis]-1 and region[2+axis] < source_size[axis]
        if low_clipped and not high_clipped:
            result[axis] -= missing
        elif high_clipped and not low_clipped:
            result[2+axis] += missing
        elif low_clipped and high_clipped:
            result[axis] -= missing/2; result[2+axis] += missing/2
    return result


def clip_box(box, source_size, last_index=True):
    """Clip to the organizer convention: right/bottom at index 3839/2159, or None if empty."""
    box = np.asarray(box, float)
    high = np.array(source_size, float)-(1. if last_index else 0.)
    clipped = np.r_[np.clip(box[:2], 0, high), np.clip(box[2:], 0, high)]
    if np.any(clipped[2:] <= clipped[:2]):
        return None
    return clipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', action='store_true', help='Fit the size prior from organizer annotations')
    parser.add_argument('--annotations', type=Path, default=Path(__file__).resolve().parents[2]/'data/drone/reference/helsinki/annotations')
    parser.add_argument('--output', type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args()
    if not args.fit:
        parser.error('Nothing to do; pass --fit')
    prior, provenance = SizePrior.fit(args.annotations)
    args.output.write_text(json.dumps({'version': 1, 'provenance': provenance,
                                       'classes': {k: {kk: {**vv} for kk, vv in v.items()} for k, v in prior.table.items()}},
                                      indent=2)+'\n')
    print(json.dumps({label: {k: round(prior.table[label][k]['intercept'], 1) for k in ('w', 'h')} for label in sorted(prior.table)}))


if __name__ == '__main__':
    main()
