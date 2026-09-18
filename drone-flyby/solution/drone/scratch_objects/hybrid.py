"""Opt-in scratch CNN, background verifier, and deterministic feature proposals."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .patch_cnn import PatchCNN,normalize_crop
from .detector import ScratchDetector,CNNSettings
from drone.template_matching.detector import TemplateDetector,iou,sha
from drone.template_matching.features import FeatureEnsemble,EnsembleSettings


def merge_proposals(cnn,features,nms_iou=.35,maximum=300):
    """Prefer geometrically verified boxes for same-class duplicate hypotheses.

    Model-family scores are not treated as calibrated to one another. Distinct
    objects/classes remain separate; the original confidence is preserved.
    """
    kept=[]
    for family,rows in [('features',features),('cnn',cnn)]:
        for row in sorted(rows,key=lambda r:-r['score']):
            if any(row['class']==k['class'] and iou(row['bbox'],k['bbox'])>nms_iou for k in kept):continue
            kept.append({**row,'method':family})
            if len(kept)>=maximum:return kept
    return kept


class PatchVerifier:
    def __init__(self,weights):
        self.checkpoint_sha256=sha(Path(weights))
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        self.classes=checkpoint['classes']
        if not self.classes or self.classes[0]!='background':raise ValueError('Verifier background class missing')
        self.model=PatchCNN(len(self.classes));self.model.load_state_dict(checkpoint['state_dict']);self.model.eval()

    def annotate(self,image,rows):
        output=[]
        for first in range(0,len(rows),64):
            batch=rows[first:first+64]
            crops=np.stack([normalize_crop(image,r['bbox']) for r in batch])
            with torch.inference_mode():
                probabilities=self.model(torch.from_numpy(crops.transpose(0,3,1,2).copy()).float()/255).softmax(1).numpy()
            for row,probabilities_row in zip(batch,probabilities):
                output.append({**row,'background_probability':float(probabilities_row[0]),
                    'verifier_probability':float(probabilities_row[self.classes.index(row['class'])]),
                    'verifier_class':self.classes[int(probabilities_row.argmax())],
                    'verifier_top_probability':float(probabilities_row.max())})
        return output


@dataclass(frozen=True)
class HybridSettings:
    cnn_threshold:float=.25
    background_threshold:float=.5
    feature_threshold:float=.8
    feature_upsample:float=2.
    feature_ransac_iterations:int=3000
    nms_iou:float=.35
    max_detections:int=300
    device:str='cpu'

    def __post_init__(self):
        if not 0<=self.background_threshold<=1:raise ValueError('Invalid background threshold')
        CNNSettings(score_threshold=self.cnn_threshold,nms_iou=self.nms_iou,max_detections=self.max_detections,device=self.device)
        EnsembleSettings(score_threshold=self.feature_threshold,upsample=self.feature_upsample,ransac_iterations=self.feature_ransac_iterations)


class HybridDetector(TemplateDetector):
    def __init__(self,weights,bank,verifier,settings=None):
        self.settings=settings or HybridSettings();cfg=self.settings
        self.cnn=ScratchDetector(weights,CNNSettings(score_threshold=cfg.cnn_threshold,nms_iou=cfg.nms_iou,max_detections=cfg.max_detections,device=cfg.device))
        self.features=FeatureEnsemble(bank,EnsembleSettings(score_threshold=cfg.feature_threshold,matching='global',upsample=cfg.feature_upsample,ransac_iterations=cfg.feature_ransac_iterations,nms_iou=cfg.nms_iou,max_detections=cfg.max_detections))
        self.verifier=PatchVerifier(verifier)
        self.checkpoint_sha256=self.cnn.checkpoint_sha256
        self.bank_sha256=sha(Path(bank)/'manifest.json')

    def detect(self,image,pixels_per_source_pixel=1.):
        cnn=self.cnn.detect(image,pixels_per_source_pixel)
        cnn=[r for r in self.verifier.annotate(image,cnn) if r['background_probability']<self.settings.background_threshold]
        features=self.features.detect(image,pixels_per_source_pixel)
        return merge_proposals(cnn,features,self.settings.nms_iou,self.settings.max_detections)

    def suppress(self,rows):
        features=self.features.suppress([r for r in rows if r.get('method')=='features'])
        cnn=[r for r in rows if r.get('method')=='cnn']
        return merge_proposals(cnn,features,self.settings.nms_iou,self.settings.max_detections)
