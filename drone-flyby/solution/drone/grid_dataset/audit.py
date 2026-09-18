"""Verify frozen pixels, split integrity, zoom pairing, and sampler eligibility."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from .build import ROOT,digest,write,render,GOOD
from .dataset import GridDataset

def audit(root):
    m=json.loads((root/'manifest.json').read_text());groups=defaultdict(set);pairs=defaultdict(list);hashsplits=defaultdict(set)
    for path,sha in m['source_hashes'].items():assert digest(ROOT/path)==sha,path
    for r in m['records']:
        p=root/r['file'];assert digest(p)==r['sha256'],p
        im=cv2.imread(str(p));assert im.shape==(384,384,3)
        assert r['native_crop_size']==[96,192,384][r['zoom']]
        pairs[r['tile_id']].append(r);hashsplits[r['sha256']].add(r['split'])
        for a in r['annotations']:
            groups[a['group']].add(r['split'])
            b=a['bbox_xyxy'];assert 0<=b[0]<b[2]<=384 and 0<=b[1]<b[3]<=384
            if r['source']=='validation':assert a['provenance'] in GOOD
        if r['source']=='validation':
            assert r['frame']<=180
            assert not r['annotation_complete']
            assert r['supervision'] in {'known_positives_only','none'}
        if r['split']=='pending_review':assert not r['eligible_for_training'] and r['background_target'] is None and not r['annotations']
    for t,rs in pairs.items():
        assert {r['zoom'] for r in rs}=={0,1,2} and len(rs)==3
        assert len({r['split'] for r in rs})==1
        assert len({str(r['annotations']) for r in rs})==1
    assert all(not {'train','dev'}<=v for v in groups.values())
    assert all(not {'train','dev'}<=v for v in hashsplits.values())
    ds=GridDataset(root)
    # Verify the real loader never treats unknown validation classes as negatives.
    for r in m['records']:
        if r['source']=='validation' and r['kind']=='positive':
            item=ds.load(r['id']);assert np.array_equal(item['known_class_mask'],item['class_targets'])
    sampled=ds.sample_batch(320)
    assert sum(ds.rows[i]['kind']=='background' for i in sampled)==160
    assert all(ds.rows[i]['eligible_for_training'] for i in sampled)
    for c,gs in ds.sampler['positive_buckets'].items():
        assert gs,c
        for g,fs in gs.items():
            for ids in fs.values():
                assert {ds.rows[i]['zoom'] for i in ids}=={0,1,2}
                assert all(ds.rows[i]['split']=='train' for i in ids)
    # Full-frame official L0 rendering versus square extraction at a real edge tile.
    r=next(r for r in m['records'] if r['source']=='reference' and r['zoom']==0)
    im=cv2.imread(str(ROOT/r['source_file']));x,y,u,v=r['source_rect_xyxy']
    low=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)[y//4:v//4,x//4:u//4]
    np.testing.assert_array_equal(cv2.resize(low,(384,384),interpolation=cv2.INTER_LINEAR),cv2.imread(str(root/r['file'])))
    report=dict(status='passed',verified_images=len(m['records']),verified_source_files=len(m['source_hashes']),paired_source_squares=len(pairs),train_dev_group_overlap=[],train_dev_exact_image_overlap=[],pending_negatives_excluded=True,partial_label_masks_verified=True,batch_foreground_background=[160,160],per_source_split={f'{s}/{split}':n for (s,split),n in Counter((r['source'],r['split']) for r in m['records']).items()},manifest_sha256=digest(root/'manifest.json'))
    write(root/'audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);audit(p.parse_args().dataset)
