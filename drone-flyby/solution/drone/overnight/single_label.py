"""Ultralytics 8.4.155 validation with predictor-compatible class selection."""
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import nms


class SingleLabelValidator(DetectionValidator):
    def postprocess(self, preds):
        outputs = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou,
            nc=0 if self.args.task == 'detect' else self.nc,
            multi_label=False,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, end2end=self.end2end,
            rotated=self.args.task == 'obb',
        )
        return [dict(bboxes=x[:, :4], conf=x[:, 4], cls=x[:, 5], extra=x[:, 6:])
                for x in outputs]
