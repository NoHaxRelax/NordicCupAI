"""Framework-neutral sampling and partial-label targets for the crop CNN.

A training loop must use known_class_mask for classification loss. Detection
background/objectness loss over spatial locations is permitted only where
annotation_complete is true. A validation positive is not a complete detection tile.
"""
import json
import random
from pathlib import Path
import cv2
import numpy as np

class GridDataset:
    def __init__(self, root, approved=False, seed=1731, combined=False):
        self.root=Path(root);self.rng=random.Random(seed)
        self.manifest=json.loads((self.root/('manifest-combined-approved.json' if combined else 'manifest-approved.json' if approved else 'manifest.json')).read_text())
        self.rows={r['id']:r for r in self.manifest['records']}
        self.sampler=json.loads((self.root/('sampler-combined-approved.json' if combined else 'sampler-approved.json' if approved else 'sampler.json')).read_text())
        self.negatives={}
        for rid in self.sampler['background_ids']:
            r=self.rows[rid]
            self.negatives.setdefault(r['source'],{}).setdefault(str(r['frame']),{}).setdefault(r['tile_id'],[]).append(rid)
        assert all(r['eligible_for_training'] for r in (self.rows[i] for i in self.sampler['background_ids']))

    def _choice(self, values):
        return self.rng.choice(list(values))

    def sample_id(self, positive=None):
        if positive is None:positive=self.rng.random()<.5
        if positive:
            groups=self._choice(self.sampler['positive_buckets'].values())
            frames=self._choice(groups.values());ids=self._choice(frames.values())
            zoom=self.rng.randrange(3)
            return self._choice([i for i in ids if self.rows[i]['zoom']==zoom])
        frames=self._choice(self.negatives.values());tiles=self._choice(frames.values());ids=self._choice(tiles.values())
        return self._choice(ids)

    def sample_batch(self, size):
        if size%2:raise ValueError('Use even batches for exact 50/50 foreground/background')
        ids=[self.sample_id(positive=i<size//2) for i in range(size)]
        self.rng.shuffle(ids);return ids

    def load(self, rid):
        r=self.rows[rid]
        if r['supervision']=='none':raise ValueError('Pending review tile has no training targets')
        image=cv2.imread(str(self.root/r['file']))
        if image is None:raise ValueError('Missing image')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        targets=np.zeros(len(self.manifest['classes']),dtype=np.float32)
        known=np.full_like(targets,float(r['annotation_complete']))
        for a in r['annotations']:
            targets[a['class_id']]=1;known[a['class_id']]=1
        return dict(image=image,class_targets=targets,known_class_mask=known,object_present=np.float32(r['background_target']==0),boxes=r['annotations'],annotation_complete=r['annotation_complete'],record=r)
