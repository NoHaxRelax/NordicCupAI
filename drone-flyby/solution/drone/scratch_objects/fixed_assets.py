"""Fixed rendered assets: context CNN, tiny silhouettes, optional local features."""
from dataclasses import dataclass
from pathlib import Path
from drone.template_matching.detector import TemplateDetector,sha
from drone.template_matching.tiny_shapes import TinyShapeDetector
from drone.template_matching.pose_pixels import PosePixelDetector
from drone.template_matching.features import FeatureEnsemble,EnsembleSettings
from .detector import ScratchDetector,CNNSettings
from .hybrid import merge_proposals,PatchVerifier


@dataclass(frozen=True)
class FixedAssetSettings:
    cnn_threshold:float=.6
    cnn_verifier_threshold:float=.7
    tiny_threshold:float=.65
    geometric:bool=True
    device:str='cpu'
    nms_iou:float=.35
    heatmap_threshold:float=.3
    heatmap_classes:tuple=('small_tower',)
    pose_pixels:bool=True
    pixel_threshold:float=.6
    calibrated_features:bool=True
    cnn_enabled:bool=True
    tiny_min_contrast:float=.2
    verify_heatmap:bool=False
    heatmap_verifier_threshold:float=.7
    calibrated_pixels:bool=False
    calibrated_pixel_threshold:float=.7
    feature_threshold:float=.8
    verify_features_below:float=0.
    weak_feature_verifier_threshold:float=.3
    weak_feature_pixel_threshold:float=.25
    verify_tiny:bool=False
    tiny_verifier_threshold:float=.8
    heatmap_enabled:bool=True
    heatmap_reclassify:bool=True

    def __post_init__(self):
        CNNSettings(score_threshold=self.cnn_threshold,nms_iou=self.nms_iou,device=self.device)
        if not all(0<=v<=1 for v in (self.heatmap_threshold,self.tiny_threshold,self.pixel_threshold,self.cnn_verifier_threshold,self.tiny_min_contrast,self.heatmap_verifier_threshold,self.calibrated_pixel_threshold)):
            raise ValueError('Invalid fixed-asset threshold')
        if not all(0<=v<=1 for v in (self.feature_threshold,self.verify_features_below,self.weak_feature_verifier_threshold,self.weak_feature_pixel_threshold,self.tiny_verifier_threshold)):
            raise ValueError('Invalid verification threshold')
        if self.verify_features_below and self.verify_features_below<self.feature_threshold:raise ValueError('Verification split must exceed proposal threshold')


class FixedAssetDetector(TemplateDetector):
    def __init__(self,weights,bank,settings=None,verifier=None,heatmap_weights=None):
        self.settings=settings or FixedAssetSettings();cfg=self.settings
        if cfg.cnn_enabled and weights is None:raise ValueError('CNN proposals require a checkpoint')
        self.cnn=ScratchDetector(weights,CNNSettings(score_threshold=cfg.cnn_threshold,device=cfg.device,nms_iou=cfg.nms_iou)) if cfg.cnn_enabled else None
        self.tiny=TinyShapeDetector(bank,cfg.tiny_threshold,cfg.tiny_min_contrast)
        self.pixels=PosePixelDetector(bank,cfg.pixel_threshold) if cfg.pose_pixels else None
        self.features=FeatureEnsemble(bank,EnsembleSettings(score_threshold=cfg.feature_threshold,strong_threshold=cfg.verify_features_below or None,nms_iou=cfg.nms_iou,calibrated_fallback=cfg.calibrated_features)) if cfg.geometric else None
        self.checkpoint_sha256=sha(Path(weights)) if weights is not None else None
        self.bank_sha256=sha(Path(bank)/'manifest.json')
        self.verifier=PatchVerifier(verifier) if verifier else None
        if (cfg.verify_heatmap or cfg.verify_tiny or cfg.verify_features_below) and not self.verifier:raise ValueError('Verification requires a verifier checkpoint')
        self.heatmap=None
        self.calibrated_pixels=None
        if cfg.calibrated_pixels:
            from drone.template_matching.calibrated_pixels import CalibratedPixelDetector
            device='cuda:'+cfg.device if cfg.device.isdigit() else cfg.device
            self.calibrated_pixels=CalibratedPixelDetector(bank,cfg.calibrated_pixel_threshold,device)
        if heatmap_weights and cfg.heatmap_enabled:
            from drone.asset_heatmap.detector import AssetHeatmapDetector
            device='cuda:'+cfg.device if cfg.device.isdigit() else cfg.device
            self.heatmap=AssetHeatmapDetector(heatmap_weights,cfg.heatmap_threshold,device)

    def suppress(self,rows):
        features=[r for r in rows if r.get('family')=='features']
        if self.features:features=self.features.suppress(features)
        tiny=[r for r in rows if r.get('family')=='tiny_shape']
        pixels=[r for r in rows if r.get('family')=='pose_pixels']
        calibrated=[r for r in rows if r.get('family')=='calibrated_pixels']
        cnn=[r for r in rows if r.get('family')=='cnn']
        heatmap=[r for r in rows if r.get('family')=='heatmap']
        deterministic=merge_proposals(pixels,merge_proposals(calibrated,features,self.settings.nms_iou),self.settings.nms_iou)
        main=merge_proposals(cnn,merge_proposals(tiny,deterministic,self.settings.nms_iou),self.settings.nms_iou)
        return merge_proposals(heatmap,main,self.settings.nms_iou)

    def detect(self,image,pixels_per_source_pixel=1.):
        cnn=self.cnn.detect(image,pixels_per_source_pixel) if self.cnn else []
        if self.verifier:cnn=[r for r in self.verifier.annotate(image,cnn) if r['background_probability']<.5 and r['verifier_probability']>=self.settings.cnn_verifier_threshold]
        rows=[{**r,'family':'cnn'} for r in cnn]
        tiny=self.tiny.detect(image,pixels_per_source_pixel)
        if self.settings.verify_tiny:tiny=[r for r in self.verifier.annotate(image,tiny) if r['verifier_class']==r['class'] and r['verifier_probability']>=self.settings.tiny_verifier_threshold]
        rows.extend({**r,'family':'tiny_shape'} for r in tiny)
        if self.pixels:rows.extend({**r,'family':'pose_pixels'} for r in self.pixels.detect(image,pixels_per_source_pixel))
        if self.features:
            features=self.features.detect(image,pixels_per_source_pixel)
            threshold=self.settings.verify_features_below
            if threshold:
                strong=[r for r in features if r['score']>=threshold];weak=[r for r in features if r['score']<threshold];verified=[]
                for r in self.verifier.annotate(image,weak):
                    if r['verifier_class']!=r['class'] or r['verifier_probability']<self.settings.weak_feature_verifier_threshold:continue
                    matcher=next(m for m in self.features.matchers if m.settings.descriptor+'-'+m.settings.geometry==r['matcher'])
                    appearance=matcher.appearance_similarity(image,r)
                    if appearance>=self.settings.weak_feature_pixel_threshold:verified.append({**r,'appearance_similarity':appearance,'verified_weak_feature':True})
                features=strong+verified
            rows.extend({**r,'family':'features'} for r in features)
        if self.calibrated_pixels:rows.extend(self.calibrated_pixels.detect(image,pixels_per_source_pixel))
        if self.heatmap:
            proposals=self.heatmap.detect(image,pixels_per_source_pixel)
            if self.settings.verify_heatmap:
                if self.settings.heatmap_reclassify:
                    proposals=[{**r,'original_class':r['class'],'class':r['verifier_class']} for r in self.verifier.annotate(image,proposals)
                               if r['verifier_class']!='background' and r['verifier_top_probability']>=self.settings.heatmap_verifier_threshold]
                else:
                    proposals=[r for r in proposals if not self.settings.heatmap_classes or r['class'] in self.settings.heatmap_classes]
                    proposals=[r for r in self.verifier.annotate(image,proposals) if r['verifier_class']==r['class'] and r['verifier_probability']>=self.settings.heatmap_verifier_threshold]
            rows.extend(r for r in proposals if not self.settings.heatmap_classes or r['class'] in self.settings.heatmap_classes)
        return self.suppress(rows)
