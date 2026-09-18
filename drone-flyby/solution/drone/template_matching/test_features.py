import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from .detector import iou,sha
from .features import FeatureDetector,FeatureSettings,FeatureEnsemble,EnsembleSettings,fuse_localizations


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        rng=np.random.default_rng(59)
        self.sprite=cv2.GaussianBlur(rng.integers(0,256,(60,80,3),dtype=np.uint8),(3,3),.7)
        self.template=np.full((100,120,3),180,np.uint8)
        self.template[20:80,20:100]=self.sprite
        mask=np.zeros((100,120),np.uint8)
        mask[20:80,20:100]=255
        cv2.imwrite(str(self.path/'template.png'),self.template)
        cv2.imwrite(str(self.path/'mask.png'),mask)
        self.manifest={'templates':[{'id':'template','class':'tank','file':'template.png',
                       'sha256':sha(self.path/'template.png'),'mask_file':'mask.png',
                       'mask_sha256':sha(self.path/'mask.png'),'source':'organizer_reference'}]}
        self.save()

    def save(self):
        (self.path/'manifest.json').write_text(json.dumps(self.manifest))

    def image(self):
        image=np.full((300,500,3),70,np.uint8)
        image[50:110,60:140]=self.sprite
        image[180:240,330:410]=self.sprite
        return image

    def test_multiple_instances_and_repeatability(self):
        detector=FeatureDetector(self.path)
        image=self.image()
        rows=detector.detect(image)
        self.assertEqual(len(rows),2)
        for box in ([40,30,160,130],[310,160,430,260]):
            self.assertGreater(max(iou(box,r['bbox']) for r in rows),.98)
        self.assertEqual(rows,detector.detect(image))

    def test_calibration_fallback_fills_gaps_without_moving_primary_boxes(self):
        self.manifest['templates'][0]['calibration']=True
        self.save()
        detector=FeatureEnsemble(self.path,EnsembleSettings(calibrated_fallback=True))
        primary={'class':'tank','bbox':[0,0,20,20],'score':.82,'template_id':'primary'}
        duplicate={**primary,'bbox':[1,1,21,21],'score':.95,'calibrated_fallback':True}
        extra={**duplicate,'bbox':[50,50,70,70]}
        rows=detector.suppress([primary,duplicate,extra])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['bbox'],[0,0,20,20])
        self.assertEqual(rows[0]['score'],.82)
        self.assertEqual(len(detector.detect(self.image())),2)

    def test_weaker_hypotheses_cannot_move_strong_localizations(self):
        detector=FeatureEnsemble(self.path,EnsembleSettings(score_threshold=.7,strong_threshold=.8))
        strong={'class':'tank','bbox':[0,0,20,20],'score':.85,'template_id':'strong'}
        weak={**strong,'bbox':[3,3,23,23],'score':.74,'template_id':'weak'}
        extra={**weak,'bbox':[50,50,70,70]}
        rows=detector.suppress([weak,strong,extra])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['bbox'],strong['bbox'])
        self.assertEqual(rows[0]['score'],strong['score'])

    def test_aligned_pixels_verify_geometric_transform(self):
        detector=FeatureDetector(self.path);im=self.image();rows=detector.detect(im)
        self.assertGreater(detector.appearance_similarity(im,rows[0]),.8)
        self.assertLess(detector.appearance_similarity(np.full_like(im,100),rows[0]),.1)

    def test_calibration_annotation_extent_is_not_padded_twice(self):
        self.manifest['templates'][0].update(source='reviewed_validation',annotation_extent=True)
        self.save()
        detector=FeatureEnsemble(self.path)
        rows=detector.detect(self.image())
        for box in ([40,30,160,130],[310,160,430,260]):
            self.assertGreater(max(iou(box,r['bbox']) for r in rows),.98)

    def test_output_annotation_extent_can_differ_from_matching_crop(self):
        self.manifest['templates'][0].update(source='reviewed_validation',annotation_extent=True,annotation_box=[10,10,110,90])
        self.save()
        rows=FeatureEnsemble(self.path).detect(self.image())
        for box in ([50,40,150,120],[320,170,420,250]):
            self.assertGreater(max(iou(box,r['bbox']) for r in rows),.98)

    def test_foreground_geometry_removes_background_margin(self):
        self.manifest['templates'][0]['source']='reviewed_validation'
        self.save()
        rows=FeatureDetector(self.path).detect(self.image())
        self.assertEqual(len(rows),2)
        for expected in ([52,44,148,116],[322,174,418,246]):
            self.assertGreater(max(iou(expected,r['bbox']) for r in rows),.94)

    def test_affine_foreshortening(self):
        matrix=np.float32([[.9,.25,70],[-.15,.75,65]])
        image=cv2.warpAffine(self.template,matrix,(320,240),borderValue=(70,70,70))
        corners=cv2.transform(np.float32([[[0,0],[120,0],[120,100],[0,100]]]),matrix)[0]
        expected=[*corners.min(0),*corners.max(0)]
        detector=FeatureDetector(self.path,FeatureSettings(descriptor='root',geometry='affine'))
        rows=detector.detect(image)
        self.assertTrue(rows)
        self.assertGreater(max(iou(expected,r['bbox']) for r in rows),.9)

    def test_background_rejection(self):
        detector=FeatureDetector(self.path)
        self.assertEqual(detector.detect(np.zeros((200,300,3),np.uint8)),[])
        noise=np.random.default_rng(91).integers(0,256,(200,300,3),dtype=np.uint8)
        self.assertEqual(detector.detect(noise),[])

    def test_ensemble_deduplicates_and_exports_coordinates(self):
        detector=FeatureEnsemble(self.path)
        rows=detector.detect(self.image())
        self.assertEqual(len(rows),2)
        predictions=detector.predict(self.image(),(0,0,500,300),(500,300))
        self.assertEqual(len(predictions),2)
        for p in predictions:
            x1,y1,x2,y2=p['bbox']
            self.assertTrue(0<=x1<x2<=1 and 0<=y1<y2<=1)
        with self.assertRaises(ValueError):
            detector.detect(self.image(),0)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            FeatureSettings(geometry='projective')
        with self.assertRaises(ValueError):
            FeatureSettings(foreground_padding=-.1)
        with self.assertRaises(ValueError):
            EnsembleSettings(nms_iou=2)

    def test_global_index_multiple_instances_and_repeatability(self):
        detector=FeatureEnsemble(self.path,EnsembleSettings(matching='global'))
        rows=detector.detect(self.image())
        self.assertEqual(len(rows),2)
        self.assertEqual(rows,detector.detect(self.image()))
        for expected in ([40,30,160,130],[310,160,430,260]):
            self.assertGreater(max(iou(expected,r['bbox']) for r in rows),.98)

    def test_fractional_upsampling_keeps_native_coordinates(self):
        detector=FeatureEnsemble(self.path,EnsembleSettings(matching='global',upsample=1.5))
        rows=detector.detect(self.image())
        self.assertEqual(len(rows),2)
        for expected in ([40,30,160,130],[310,160,430,260]):
            self.assertGreater(max(iou(expected,r['bbox']) for r in rows),.95)
        with self.assertRaises(ValueError):EnsembleSettings(upsample=0)

    def test_downsampled_template_coordinate_mapping(self):
        detector=FeatureDetector(self.path,FeatureSettings(template_scales=(.5,)))
        small=cv2.resize(self.sprite,(40,30),interpolation=cv2.INTER_AREA)
        image=np.full((150,200,3),70,np.uint8)
        image[40:70,60:100]=small
        rows=detector.detect(image,.5)
        self.assertTrue(rows)
        self.assertGreater(max(iou([50,30,110,80],r['bbox']) for r in rows),.95)

    def test_fusion_handles_nested_extents_without_merging_neighbors(self):
        rows=[{'class':'tank','bbox':b,'score':s,'template_id':str(i),'reprojection_error':e}
              for i,(b,s,e) in enumerate([([0,0,20,40],.91,1),([0,16,16,36],.90,.3),([30,16,46,36],.88,.2)])]
        fused=fuse_localizations(rows)
        self.assertEqual(len(fused),2)
        self.assertEqual(fused[0]['fused_hypotheses'],2)
        self.assertEqual(fused[0]['score'],.91)
        self.assertGreater(iou(fused[0]['bbox'],[0,10,20,36]),.65)


if __name__=='__main__':
    unittest.main()
