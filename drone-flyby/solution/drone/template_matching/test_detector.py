import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from .detector import Settings, TemplateDetector, iou, nms, sha
from .evaluate import measure, validation_truth


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        rng = np.random.default_rng(42)
        self.patch = rng.integers(20, 230, (24,32,3), dtype=np.uint8)
        cv2.imwrite(str(self.path/'sprite.png'), self.patch)
        (self.path/'manifest.json').write_text(json.dumps({'templates': [
            {'id': 'sprite', 'file': 'sprite.png', 'sha256': sha(self.path/'sprite.png'), 'class': 'tank'}]}))

    def detector(self, **kwargs):
        return TemplateDetector(self.path, Settings(scales=(1.,), angles=(0.,), **kwargs))

    def test_blind_discovery_two_instances_and_coordinates(self):
        image = np.full((140,240,3), 70, np.uint8)
        image[37:61,55:87] = self.patch
        image[83:107,161:193] = self.patch
        detector = self.detector()
        predictions = detector.detect(image)
        self.assertEqual(len(predictions), 2)
        for box in ([55,37,87,61], [161,83,193,107]):
            self.assertTrue(any(iou(box,p['bbox']) > .99 for p in predictions))
        normalized = detector.predict(image, (100,200,340,340), (500,500))
        self.assertTrue(any(np.allclose(p['bbox'], [.31,.474,.374,.522]) for p in normalized))
        self.assertEqual(predictions, detector.detect(image))
        from drone.perspective_tracking.motion import ViewGeometry
        view = ViewGeometry((500,500),(100,200,340,340),(240,140))
        tracked = detector.tracking_detections(image,view)
        self.assertEqual(len(tracked),2)
        self.assertEqual(tracked[0].label,'tank')
        self.assertEqual(tracked[0].box,tuple(predictions[0]['bbox']))

    def test_blank_and_unrelated_background_rejected(self):
        self.assertEqual(self.detector().detect(np.zeros((100,200,3),np.uint8)), [])
        noise = np.random.default_rng(54).integers(0,255,(100,200,3),dtype=np.uint8)
        self.assertEqual(self.detector().detect(noise), [])

    def test_delivered_half_scale(self):
        small = cv2.resize(self.patch, (16,12), interpolation=cv2.INTER_AREA)
        image = np.full((60,100,3),70,np.uint8)
        image[22:34,31:47] = small
        predictions = self.detector().detect(image,.5)
        self.assertTrue(any(iou([31,22,47,34], p['bbox']) > .99 for p in predictions))

    def test_rotation_and_scale(self):
        detector = TemplateDetector(self.path, Settings(scales=(1.18,), angles=(12.,)))
        patch = detector.variants(1.)[0][3]
        h,w = patch.shape[:2]
        image = np.full((100,200,3),70,np.uint8)
        image[30:30+h,41:41+w] = patch
        self.assertTrue(any(iou([41,30,41+w,30+h],p['bbox'])>.99 for p in detector.detect(image)))

    def test_corrupt_bank_rejected(self):
        (self.path/'sprite.png').write_bytes(b'broken')
        with self.assertRaises(ValueError):
            self.detector()

    def test_masked_bank_background_transfer(self):
        # A textured sprite on a different surrounding background must remain detectable.
        rng = np.random.default_rng(93)
        sprite = rng.integers(20,200,(20,20,3),dtype=np.uint8)
        template = np.full((30,30,3),120,np.uint8)
        template[5:25,5:25] = sprite
        cv2.imwrite(str(self.path/'sprite.png'),template)
        mask = np.zeros((30,30),np.uint8)
        mask[4:26,4:26] = 255
        cv2.imwrite(str(self.path/'mask.png'),mask)
        m = json.loads((self.path/'manifest.json').read_text())
        m['templates'][0].update(sha256=sha(self.path/'sprite.png'),mask_file='mask.png',mask_sha256=sha(self.path/'mask.png'))
        (self.path/'manifest.json').write_text(json.dumps(m))
        image = np.full((100,200,3),170,np.uint8)
        image[35:55,65:85] = sprite
        detector = self.detector()
        rows = detector.detect(image)
        self.assertTrue(any(iou([60,30,90,60],r['bbox'])>.99 for r in rows))
        self.assertEqual(detector.detect(np.zeros_like(image)),[])
        (self.path/'mask.png').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            self.detector()

    def test_classwise_nms_preserves_distinct_instances(self):
        rows=[{'class':'tank','bbox':b,'score':s,'template_id':'a'} for b,s in
              [([0,0,20,20],.9),([1,1,21,21],.8),([50,50,70,70],.85)]]
        rows.append({'class':'hangar','bbox':[0,0,20,20],'score':.75,'template_id':'b'})
        self.assertEqual(len(nms(rows)),3)

    def test_one_to_one_scoring(self):
        truth = [{'class':'tank','bbox':[1,1,11,11]}, {'class':'tank','bbox':[2,2,12,12]}]
        predictions = [{'class':'tank','bbox':[1,1,11,11],'score':1}]
        self.assertEqual(measure(predictions,truth)['matched'],1)

    def test_reviewed_truth_excludes_projection_and_resolves_alias(self):
        config=self.path/'drone/overnight/config.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'track_aliases':{'fragment':'physical-object'}}))
        directory=self.path/'data/drone/training/algorithmic-full-validation'
        directory.mkdir(parents=True)
        good={'frame':100,'class':'tank','bbox_source_xyxy':[10,10,30,30],
              'provenance':'reviewed_positive','review_status':'directly_reviewed_track'}
        projected={**good,'bbox_source_xyxy':[40,40,60,60],'provenance':'algorithmic_shared_motion'}
        clipped={**good,'bbox_source_xyxy':[0,10,30,30]}
        right_clipped={**good,'bbox_source_xyxy':[3800,10,3839,30]}
        bottom_clipped={**good,'bbox_source_xyxy':[10,2100,30,2159]}
        (directory/'fragment.json').write_text(json.dumps({'track_id':'fragment','annotations':[good,projected,clipped,right_clipped,bottom_clipped,good]}))
        truth,hashes=validation_truth(self.path,[100])
        self.assertEqual(len(truth[100]),1)
        self.assertEqual(truth[100][0]['track'],'physical-object')
        self.assertEqual(len(hashes),1)

    def test_small_view_and_invalid_scales(self):
        self.assertEqual(self.detector().detect(np.zeros((4,4,3),np.uint8)),[])
        with self.assertRaises(ValueError):
            Settings(scales=(0.,))
        with self.assertRaises(ValueError):
            self.detector().detect(np.zeros((40,40,3),np.uint8),float('nan'))


if __name__ == '__main__':
    unittest.main()
