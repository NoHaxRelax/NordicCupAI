"""Opt-in CNN adapter with the same image/source-coordinate API as templates."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from drone.template_matching.detector import TemplateDetector,sha


@dataclass(frozen=True)
class CNNSettings:
    score_threshold:float=.25
    nms_iou:float=.35
    max_detections:int=300
    image_size:int=960
    device:str='cpu'

    def __post_init__(self):
        if not 0<=self.score_threshold<=1 or not 0<=self.nms_iou<=1:
            raise ValueError('Thresholds must be in [0,1]')
        if self.max_detections<1 or self.image_size<32:
            raise ValueError('Invalid detector dimensions/limit')


class ScratchDetector(TemplateDetector):
    """Only instantiate checkpoints produced by this trusted local experiment."""
    def __init__(self,weights,settings=None,expected_sha256=None):
        self.settings=settings or CNNSettings()
        weights=Path(weights)
        self.checkpoint_sha256=sha(weights)
        if expected_sha256 and self.checkpoint_sha256!=expected_sha256:
            raise ValueError('Checkpoint hash mismatch')
        from ultralytics import YOLO
        self.model=YOLO(str(weights))

    def detect(self,image,pixels_per_source_pixel=1.):
        if image is None or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected uint8 BGR image')
        if not np.isfinite(pixels_per_source_pixel) or pixels_per_source_pixel<=0:
            raise ValueError('Invalid image scale')
        cfg=self.settings
        result=self.model.predict(image,imgsz=cfg.image_size,conf=cfg.score_threshold,iou=cfg.nms_iou,
                                  max_det=cfg.max_detections,device=cfg.device,verbose=False)[0]
        return [{'class':self.model.names[int(c)],'bbox':box,'score':float(score)}
                for box,c,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())]
