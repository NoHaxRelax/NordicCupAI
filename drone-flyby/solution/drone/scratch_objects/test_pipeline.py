import unittest
import numpy as np
from .prepare import transform, overlap
from .evaluate import measure,nms,starts

class PipelineTests(unittest.TestCase):
    def test_rotation_keeps_visible_mask(self):
        image=np.zeros((20,40,3),np.uint8);mask=np.full((20,40),255,np.uint8)
        patch,alpha=transform(image,mask,90,2)
        self.assertEqual(patch.shape[:2],(80,41)) # ceil accounts for floating cosine
        self.assertGreater(np.count_nonzero(alpha),3000)
    def test_one_to_one_and_confusion(self):
        truth=[{'class':'tank','bbox':[0,0,10,10]}]
        rows=[{'class':'plane','bbox':[0,0,10,10],'score':.9}, {'class':'tank','bbox':[0,0,10,10],'score':.8}]
        self.assertEqual(measure(rows,truth)['matched'],1)
        self.assertEqual(measure(rows[:1],truth)['matched'],0)
        self.assertEqual(measure(rows[:1],truth,False)['matched'],1)
        self.assertEqual(measure(rows*2,truth)['matched'],1)
    def test_hybrid_keeps_distinct_instances_and_feature_priority(self):
        from .hybrid import merge_proposals
        feature={'class':'tank','bbox':[0,0,10,10],'score':.8}
        cnn=[{'class':'tank','bbox':[1,0,11,10],'score':.99}, {'class':'tank','bbox':[20,0,30,10],'score':.9}]
        merged=merge_proposals(cnn,[feature])
        self.assertEqual(len(merged),2)
        self.assertEqual(merged[0]['bbox'],feature['bbox'])
        self.assertEqual(merged[0]['score'],.8)
        self.assertEqual(merged[1]['bbox'],[20,0,30,10])
        self.assertNotIn('method',feature)

    def test_verifier_thresholds_never_promote_rejected_proposals(self):
        from .verify_evaluation import filter_predictions
        row={'class':'tank','bbox':[0,0,10,10],'score':.2,'verifier_probability':.9,'background_probability':.01,'verifier_class':'tank','verifier_top_probability':.9}
        self.assertEqual(filter_predictions([row],.25,'class_gate',.5),[])
        self.assertEqual(filter_predictions([row],.1,'class_gate',.5),[row])
        background={**row,'score':.99,'background_probability':.99,'verifier_class':'background','verifier_top_probability':.99}
        self.assertEqual(filter_predictions([background],.1,'background_gate',.5),[])
        self.assertEqual(filter_predictions([background],.1,'reclassify',.5),[])
        alternate={**row,'verifier_class':'small_tower'}
        changed=filter_predictions([alternate],.1,'reclassify',.5)[0]
        self.assertEqual(changed['class'],'small_tower')
        self.assertEqual(changed['original_class'],'tank')
        self.assertEqual(alternate['class'],'tank')

    def test_pixel_refinement_preserves_confidence_and_rejects_blank(self):
        import tempfile,json,cv2
        from pathlib import Path
        from .prepare import sha
        from .refine import PixelRefiner
        with tempfile.TemporaryDirectory() as tmp:
            bank=Path(tmp)
            patch=np.full((32,32,3),(140,170,190),np.uint8)
            patch[10:22,10:22]=(80,150,90)
            mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255
            cv2.imwrite(str(bank/'image.png'),patch);cv2.imwrite(str(bank/'mask.png'),mask)
            (bank/'manifest.json').write_text(json.dumps({'templates':[{'class':'small_launcher','file':'image.png','sha256':sha(bank/'image.png'),'mask_file':'mask.png','mask_sha256':sha(bank/'mask.png')}]}))
            refiner=PixelRefiner(bank)
            image=np.full((48,48,3),(140,170,190),np.uint8);image[20:30,20:30]=(80,150,90)
            rows=[{'class':'small_launcher','bbox':[8,8,42,42],'score':.6}]
            refined=refiner.refine(image,rows)[0]
            self.assertEqual(refined['bbox'],[18.,18.,32.,32.])
            self.assertEqual(refined['score'],.6)
            self.assertEqual(refined['unrefined_bbox'],rows[0]['bbox'])
            image[:]=(140,170,190)
            self.assertEqual(refiner.refine(image,rows),rows)

    def test_cnn_source_coordinate_adapter(self):
        from .detector import ScratchDetector,CNNSettings
        class Tensor:
            def __init__(self,value):self.value=value
            def cpu(self):return self
            def tolist(self):return self.value
        class Boxes:
            xyxy=Tensor([[10,20,30,40]]);cls=Tensor([0]);conf=Tensor([.9])
        class Result:boxes=Boxes()
        class Model:
            names={0:'tank'}
            def predict(self,*args,**kwargs):return [Result()]
        detector=ScratchDetector.__new__(ScratchDetector)
        detector.settings=CNNSettings();detector.model=Model()
        image=np.zeros((100,200,3),np.uint8)
        rows=detector.predict(image,(100,200,500,400),(1000,1000))
        self.assertTrue(np.allclose(rows[0]['bbox'],[.12,.24,.16,.28]))
        self.assertEqual(rows[0]['object_id'],'tank')
        with self.assertRaises(ValueError):detector.detect(image,-1)

    def test_tile_coverage_and_class_nms(self):
        self.assertEqual(starts(3840,960),[0,720,1440,2160,2880])
        rows=[{'class':c,'bbox':[0,0,10,10],'score':s} for c,s in [('tank',.9),('tank',.8),('plane',.7)]]
        self.assertEqual(len(nms(rows)),2)
        self.assertEqual(overlap([0,0,10,10],[10,10,20,20]),0)

    def test_bundle_rejects_changed_checkpoint_before_model_loading(self):
        import tempfile,json
        from pathlib import Path
        from .bundle import load_bundle
        from .prepare import sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weights=root/'weights.pt';weights.write_bytes(b'original')
            manifest=root/'bundle.json'
            manifest.write_text(json.dumps(dict(format='fixed-asset-detector-v1',files={
                'weights':dict(path='weights.pt',sha256=sha(weights))})))
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Bundle checksum mismatch: weights'):
                load_bundle(manifest)

    def test_verification_only_bundle_does_not_require_disabled_models(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from .bundle import load_bundle
        from .prepare import sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'verifier.pt').write_bytes(b'verifier');(root/'bank').mkdir();(root/'bank/manifest.json').write_text('{}')
            manifest=root/'bundle.json';manifest.write_text(json.dumps(dict(format='fixed-asset-detector-v2',files={'verifier':dict(path='verifier.pt',sha256=sha(root/'verifier.pt')),'bank':dict(path='bank',sha256=sha(root/'bank/manifest.json'))},settings=dict(cnn_enabled=False,heatmap_enabled=False,heatmap_classes=[]))))
            with patch('drone.scratch_objects.bundle.FixedAssetDetector') as model:
                load_bundle(manifest)
                self.assertIsNone(model.call_args.args[0])
                self.assertIsNone(model.call_args.args[4])

    def test_fixed_asset_heatmap_verifier_reclassifies_and_rejects_background(self):
        from .fixed_assets import FixedAssetDetector,FixedAssetSettings
        class Empty:
            def detect(self,*args):return []
        class Heatmap:
            def detect(self,*args):return [dict(class_name='unused',**{'class':'tank'},bbox=[0,0,10,10],score=.2,family='heatmap')]
        class Verifier:
            def annotate(self,image,rows):return [{**r,'verifier_class':c,'verifier_top_probability':p} for r in rows for c,p in [('small_tower',.95),('background',.99),('tank',.2)]]
        detector=FixedAssetDetector.__new__(FixedAssetDetector)
        detector.settings=FixedAssetSettings(cnn_enabled=False,heatmap_classes=(),verify_heatmap=True,heatmap_verifier_threshold=.9)
        detector.calibrated_pixels=None;detector.cnn=None;detector.features=None;detector.pixels=None;detector.tiny=Empty();detector.heatmap=Heatmap();detector.verifier=Verifier()
        result=detector.detect(np.zeros((20,20,3),np.uint8))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['class'],'small_tower')
        self.assertEqual(result[0]['original_class'],'tank')
        self.assertEqual(result[0]['score'],.2)

    def test_proposal_training_crop_tracks_foreground_extent(self):
        from drone.asset_heatmap.patch_train import foreground_box
        rng=np.random.default_rng(18);mask=np.zeros((64,64),np.uint8);mask[20:35,16:46]=255
        for _ in range(20):
            x1,y1,x2,y2=foreground_box(mask,rng)
            self.assertTrue(0<=x1<x2<=64 and 0<=y1<y2<=64)
            self.assertGreater((46-16)/(x2-x1),.49)
            self.assertLess((46-16)/(x2-x1),1.01)

    def test_weak_geometry_requires_both_pixel_and_cnn_agreement(self):
        from types import SimpleNamespace
        from .fixed_assets import FixedAssetDetector,FixedAssetSettings
        class Empty:
            def detect(self,*args):return []
        class Matcher:
            settings=SimpleNamespace(descriptor='sift',geometry='similarity')
            def appearance_similarity(self,image,row):return row['pixel_test']
        class Features:
            matchers=[Matcher()]
            def detect(self,*args):
                return [dict(**{'class':'tank'},bbox=[i*20,0,i*20+10,10],score=score,template_id=str(i),matcher='sift-similarity',pixel_test=pixel,prob_test=prob) for i,(score,pixel,prob) in enumerate([(.9,0,0),(.74,.3,.4),(.74,.1,.99),(.74,.9,.1)])]
            def suppress(self,rows):return rows
        class Verifier:
            def annotate(self,image,rows):return [{**r,'verifier_class':'tank','verifier_probability':r['prob_test']} for r in rows]
        detector=FixedAssetDetector.__new__(FixedAssetDetector);detector.settings=FixedAssetSettings(cnn_enabled=False,feature_threshold=.7,verify_features_below=.8)
        detector.cnn=None;detector.heatmap=None;detector.pixels=None;detector.calibrated_pixels=None;detector.tiny=Empty();detector.features=Features();detector.verifier=Verifier()
        rows=detector.detect(np.zeros((20,100,3),np.uint8))
        self.assertEqual({r['template_id']for r in rows},{'0','1'})
        self.assertTrue(next(r for r in rows if r['template_id']=='1')['verified_weak_feature'])

if __name__=='__main__':unittest.main()
