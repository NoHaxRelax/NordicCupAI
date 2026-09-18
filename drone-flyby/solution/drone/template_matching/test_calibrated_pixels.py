import unittest
import cv2,numpy as np,torch
from .calibrated_pixels import masked_ncc_fft

class FFTTests(unittest.TestCase):
 def test_complete_detector_finds_blind_translation_and_rejects_blank(self):
  import json,tempfile
  from pathlib import Path
  from .calibrated_pixels import CalibratedPixelDetector
  from .detector import sha,iou
  with tempfile.TemporaryDirectory() as directory:
   bank=Path(directory);rng=np.random.default_rng(52);patch=rng.integers(30,220,(20,28,3),dtype=np.uint8);mask=np.full((20,28),255,np.uint8)
   cv2.imwrite(str(bank/'crop.png'),patch);cv2.imwrite(str(bank/'mask.png'),mask)
   (bank/'manifest.json').write_text(json.dumps(dict(templates=[dict(id='asset',**{'class':'tank'},file='crop.png',sha256=sha(bank/'crop.png'),mask_file='mask.png',mask_sha256=sha(bank/'mask.png'),calibration=True)])))
   model=CalibratedPixelDetector(bank,.8);image=np.full((70,90,3),100,np.uint8);image[31:51,43:71]=patch
   rows=model.detect(image);self.assertTrue(any(iou(r['bbox'],[43,31,71,51])>.99 and r['score']>.9 for r in rows))
   self.assertEqual(model.detect(np.full_like(image,100)),[])

 def test_valid_masked_correlation_matches_opencv(self):
  rng=np.random.default_rng(44);scene=rng.uniform(-.5,.5,(2,45,61)).astype(np.float32);patch=scene[:,17:28,24:37].copy();mask=(rng.random((11,13))>.3).astype(np.float32)
  maps=masked_ncc_fft(torch.from_numpy(scene),torch.from_numpy(patch[None]),torch.from_numpy(mask[None,None])).numpy()[0]
  for c in range(2):
   reference=cv2.matchTemplate(scene[c],patch[c],cv2.TM_CCOEFF_NORMED,mask=mask)
   np.testing.assert_allclose(maps[c],reference,atol=2e-5)
   self.assertEqual(np.unravel_index(maps[c].argmax(),maps[c].shape),(17,24))
 def test_blank_has_no_nan_or_positive_similarity(self):
  scene=torch.zeros((2,40,60));patch=torch.rand((1,2,9,13));mask=torch.ones((1,1,9,13));result=masked_ncc_fft(scene,patch,mask)
  self.assertTrue(torch.isfinite(result).all());self.assertEqual(float(result.max()),0)
if __name__=='__main__':unittest.main()
