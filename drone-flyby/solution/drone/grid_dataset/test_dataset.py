import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from .build import SIDE, positions, render, contains
from .dataset import GridDataset

class GeometryTests(unittest.TestCase):
    def test_zoom_render_matches_delivered_view(self):
        im=np.random.default_rng(1).integers(0,256,(2160,3840,3),dtype=np.uint8)
        for x,y in [(192,384),(3456,1776)]:
            raw=im[y:y+384,x:x+384]
            for z,divisor in enumerate([4,2,1]):
                delivered=cv2.resize(im,(3840//divisor,2160//divisor),interpolation=cv2.INTER_AREA)
                crop=delivered[y//divisor:(y+384)//divisor,x//divisor:(x+384)//divisor]
                expected=cv2.resize(crop,(384,384),interpolation=cv2.INTER_LINEAR) if divisor>1 else crop
                np.testing.assert_array_equal(render(raw,z),expected)
    def test_overlap_and_edges(self):
        for length in [3840,2160]:
            ps=positions(length)
            self.assertEqual(ps[0],0);self.assertEqual(ps[-1]+SIDE,length)
            self.assertLessEqual(max(b-a for a,b in zip(ps,ps[1:])),192)
            for start in range(length-192+1):
                self.assertTrue(any(p<=start and start+192<=p+SIDE for p in ps))

if __name__=='__main__':unittest.main()

class ApprovalTests(unittest.TestCase):
    def fixture(self, path):
        import json
        from .build import write,digest
        for z in range(3):
            (path/f'image-L{z}.png').write_bytes(b'synthetic-test-pixels-'+bytes([z]))
        rows=[dict(tile_id='synthetic',split='pending_review',annotations=[],file=f'image-L{z}.png',sha256=digest(path/f'image-L{z}.png')) for z in range(3)]
        write(path/'manifest.json',dict(records=rows));write(path/'sampler.json',dict(background_ids=[]))
        c=dict(tile_id='synthetic',frame=5,source_rect_xyxy=[0,0,384,384],source_sha256='synthetic-only',native_image='image-L2.png',native_image_sha256=digest(path/'image-L2.png'))
        write(path/'review-candidates.json',dict(candidates=[c]))
        return dict(reviewer='Oscar Svendsen',attested_personal_visual_review=True,dataset_manifest_sha256=digest(path/'manifest.json'),decisions=[dict(c,status='verified_empty')])

    def test_approve_exact_pairs_and_preserve_original(self):
        from .build import write,digest
        from .approve import approve
        import json
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);receipt=self.fixture(path);original=digest(path/'manifest.json');write(path/'receipt.json',receipt)
            # Add IDs used by the approved sampler; all files are synthetic fixtures.
            m=json.loads((path/'manifest.json').read_text())
            for z,r in enumerate(m['records']):r['id']=f'synthetic-L{z}'
            write(path/'manifest.json',m);original=digest(path/'manifest.json');receipt['dataset_manifest_sha256']=original;write(path/'receipt.json',receipt)
            approve(path,path/'receipt.json')
            self.assertEqual(digest(path/'manifest.json'),original)
            updated=json.loads((path/'manifest-approved.json').read_text())
            self.assertTrue(all(r['eligible_for_training'] and r['annotation_complete'] for r in updated['records']))

    def test_reject_changed_pixels_and_wrong_dataset(self):
        from .build import write
        from .approve import approve
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);receipt=self.fixture(path);write(path/'receipt.json',receipt)
            (path/'image-L2.png').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'pixels changed'):approve(path,path/'receipt.json')
            receipt['dataset_manifest_sha256']='wrong';write(path/'receipt.json',receipt)
            with self.assertRaisesRegex(ValueError,'different dataset'):approve(path,path/'receipt.json')
