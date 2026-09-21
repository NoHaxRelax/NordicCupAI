"""N-member ensemble detector for the live endpoint: ``CP02_DRONE_DETECTOR=ensemble_detector:build``.

Every member sees every delivered view; all boxes are pooled and de-duplicated with class-wise NMS. No class routing and
no level routing by default (Oscar's generalisation rule: nothing here is fitted to the validation flight).

Member kinds
  tile    256 px tile YOLO: the view is upsampled (x4 at L0, x2 at L1, x1 at L2) and predicted at full width.
  views   1280 px views YOLO (Elias-style recipe): the view as delivered, letterboxed to ``imgsz``.
  frcnn   torchvision Faster R-CNN v2 checkpoint from train_frcnn.py (class order from the checkpoint or ``names``).

Confidence calibration (optional, per member): a monotone piecewise-linear map {"x": [...], "y": [...]} applied to the
member's confidences before pooling, so members whose scores live on different scales can be ranked together. Fit it with
fit_calibration.py on TRAIN-side data only, never on the validation flight.

Environment (DRONE_* names; the checkpoint-02 server maps CP02_DRONE_* onto them):
  DRONE_ENS_MEMBERS         JSON list (or the path of a JSON file holding it), e.g.
                            [{"kind":"tile","weights":"/w/m-fly-epoch40-tile.pt"},
                             {"kind":"tile","weights":"/w/m-rows.pt"},
                             {"kind":"views","weights":"/w/fly36-l.pt","imgsz":1280,"calibration":"/w/fly36-l.cal.json"}]
                            optional per member: "scales" [L0,L1,L2] (tile), "imgsz" (views), "names" (frcnn),
                            "calibration" (path), "levels" [0,1,2] (default all), "name"
  DRONE_ENS_TILE_WEIGHTS / DRONE_ENS_VIEWS_WEIGHTS   two-member shorthand (the 0.7814 validation-API configuration)
  DRONE_ENS_TILE_BOX_SCALE  JSON {class: [scale_w, scale_h]} applied about the box centre to TILE-member boxes before the
                            merge, e.g. {"medium_launcher": [0.855, 0.806]}. Constants must come from ORGANISER TRAIN
                            labels (artifacts/model-sweep-20260920/BOX-SIZES.md), never from validation numbers. A member
                            may carry its own "box_scale" in the members file, which overrides this; a member retrained
                            with corrected labels should carry {} so nothing is applied to it. Default: none.
  DRONE_ENS_VIEWS_EXTENT    1 = when a tile-member box and a views-member box of the SAME class overlap at IoU >= 0.5, emit
                            one box with the views member's extent and the higher of the two confidences; a box only
                            one member fires is left untouched. No class list, no constants. Reason (organiser's Helsinki
                            frames and labels, artifacts/boxgeom-20260920/FINDINGS.md): the views member draws the
                            organiser's box convention (0.99 w / 0.98 h), the tile member is loose (1.08 / 1.13). [0]
  DRONE_SYNTH_CONF / DRONE_SYNTH_IOU / DRONE_SYNTH_HALF / DRONE_SYNTH_MAXDET / DRONE_DEVICE / DRONE_SYNTH_CLASS_CONF /
  DRONE_EXTENT_CLASS_POLICY   as in synth_detector
"""
import json
import os
import sys
import time
from contextlib import redirect_stdout

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

try:
    from dtos import OBJECT_CLASSES
except ImportError:  # outside the endpoint (local scorer)
    OBJECT_CLASSES = None

try:
    from synth_detector import install_class_extent_policy
except ImportError:
    def install_class_extent_policy(environ=os.environ):
        return None


def scale_box(box, sw, sh):
    """Scale a box about its own centre."""
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0+x1)/2., (y0+y1)/2., (x1-x0)*sw/2., (y1-y0)*sh/2.
    return [cx-w, cy-h, cx+w, cy+h]


class Calibration:
    """Monotone piecewise-linear confidence map; identity when no file is given."""

    def __init__(self, path=None):
        self.x = self.y = None
        if path:
            d = json.load(open(path))
            self.x, self.y = np.asarray(d['x'], float), np.asarray(d['y'], float)

    def __call__(self, scores):
        return scores if self.x is None else np.interp(scores, self.x, self.y).tolist()


class YoloMember:
    def __init__(self, spec, device, half):
        from ultralytics import YOLO
        with redirect_stdout(sys.stderr):
            self.model = YOLO(str(spec['weights']))
        self.kind = spec['kind']
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        self.scales = dict(zip((0, 1, 2), [float(v) for v in spec.get('scales', (4, 2, 1))]))
        self.imgsz = int(spec.get('imgsz', 1280))
        self.device, self.half = device, half

    def predict(self, image, level, conf, iou, max_det):
        if self.kind == 'tile':
            f = self.scales.get(level, 1.)
            if f != 1:
                image = cv2.resize(image, (int(round(image.shape[1]*f)), int(round(image.shape[0]*f))), interpolation=cv2.INTER_LINEAR)
            imgsz = int(image.shape[1])
        else:
            f, imgsz = 1., self.imgsz
        r = self.model.predict(image, imgsz=imgsz, conf=conf, iou=iou, device=self.device, half=self.half, max_det=max_det, verbose=False)[0]
        return ([self.names[int(c)] for c in r.boxes.cls.cpu().tolist()], [[float(v)/f for v in b] for b in r.boxes.xyxy.cpu().tolist()],
                r.boxes.conf.cpu().tolist())


class FrcnnMember:
    def __init__(self, spec, device, half):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        ck = torch.load(spec['weights'], map_location='cpu', weights_only=False)
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=ck['min_size'], max_size=ck['max_size'],
                                           box_detections_per_img=300, box_score_thresh=0.01)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, ck['ncls']+1)
        model.load_state_dict(ck['model'])
        self.model = model.to(device).eval()
        self.names = list(spec.get('names') or ck.get('names') or [])
        if not self.names:
            raise ValueError('frcnn member needs "names" (the class order of its training COCO json)')
        self.kind, self.device, self.half = 'frcnn', device, half

    def predict(self, image, level, conf, iou, max_det):
        t = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255.).to(self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
            r = self.model([t])[0]
        return ([self.names[int(c)-1] for c in r['labels'].cpu().tolist()], r['boxes'].float().cpu().tolist(), r['scores'].float().cpu().tolist())


class EnsembleDetector:
    name = 'ensemble'

    EXTENT_IOU = 0.5  # fixed up front (the usual match threshold); not tuned on any flight

    def __init__(self, members, *, device='cuda:0', conf=.25, iou=.6, half=True, max_det=300, class_conf=None, warmup=True,
                 views_extent=False, tile_box_scale=None):
        tile_box_scale = tile_box_scale or {}
        threads = cv2.getNumThreads()
        try:
            self.members = []
            for i, spec in enumerate(members):
                member = (FrcnnMember if spec['kind'] == 'frcnn' else YoloMember)(spec, device, half)
                if OBJECT_CLASSES is not None and set(member.names) != set(OBJECT_CLASSES):
                    raise ValueError(f"member {i} classes {member.names} are not the competition classes")
                member.label = spec.get('name') or f"{spec['kind']}{i}"
                member.calibration = Calibration(spec.get('calibration'))
                member.levels = set(int(v) for v in spec['levels']) if spec.get('levels') is not None else None
                member.box_scale = {k: (float(v[0]), float(v[1])) for k, v in (spec.get('box_scale') if spec.get('box_scale') is not None
                                                                              else (tile_box_scale if member.kind == 'tile' else {})).items()}
                self.members.append(member)
        finally:
            cv2.setNumThreads(threads)
        if not self.members:
            raise ValueError('an ensemble needs at least one member')
        self.labels = sorted({n for m in self.members for n in m.names})  # members may order classes differently: pool by NAME
        self.label_id = {n: i for i, n in enumerate(self.labels)}
        self.device, self.conf, self.iou, self.half, self.max_det = device, float(conf), float(iou), bool(half), int(max_det)
        self.class_conf = {k: float(v) for k, v in (class_conf or {}).items()}
        self.views_extent = bool(views_extent)
        self.extent_swaps = 0
        self.member_ms = {m.label: [] for m in self.members}
        if warmup:
            for _ in range(3):
                for level in (0, 1, 2):
                    self(np.zeros((540, 960, 3), np.uint8), {'view': {'resolution_level': level}})
            self.member_ms = {m.label: [] for m in self.members}

    def __call__(self, image, request, _image=None):
        image = _image if _image is not None else image
        level = int((request.get('view') or {}).get('resolution_level', 2))
        names, boxes, scores, kinds = [], [], [], []
        with redirect_stdout(sys.stderr):
            for m in self.members:
                if m.levels is not None and level not in m.levels:
                    continue
                t = time.perf_counter()
                # the floor is applied to the member's RAW confidence, so calibration never changes what a member emits
                n, b, s = m.predict(image, level, self.conf, self.iou, self.max_det)
                keep = [i for i, v in enumerate(s) if v >= self.conf]
                n, b, s = [n[i] for i in keep], [b[i] for i in keep], m.calibration([s[i] for i in keep])
                if m.box_scale:
                    b = [scale_box(box, *m.box_scale[cls]) if cls in m.box_scale else box for cls, box in zip(n, b)]
                names += n; boxes += b; scores += list(s); kinds += [m.kind]*len(n)
                self.member_ms[m.label].append((time.perf_counter()-t)*1000)
        if not boxes:
            return []
        if self.views_extent:
            self._take_views_extent(names, boxes, scores, kinds)
        keep = batched_nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),
                           torch.tensor([self.label_id[n] for n in names]), self.iou).tolist()
        return [{'label': names[i], 'box': boxes[i], 'confidence': float(scores[i])} for i in keep
                if scores[i] >= self.class_conf.get(names[i], 0.)]


    def _take_views_extent(self, names, boxes, scores, kinds):
        """In place: a tile box matched by a same-class views box takes that box's extent and the higher confidence."""
        views = [i for i, k in enumerate(kinds) if k == 'views']
        if not views:
            return
        vb = torch.tensor([boxes[i] for i in views], dtype=torch.float32)
        for i, kind in enumerate(kinds):
            if kind != 'tile':
                continue
            same = [j for j, v in enumerate(views) if names[v] == names[i]]
            if not same:
                continue
            overlap = box_iou(torch.tensor([boxes[i]], dtype=torch.float32), vb[same])[0]
            best = int(overlap.argmax())
            if float(overlap[best]) >= self.EXTENT_IOU:
                v = views[same[best]]
                boxes[i] = list(boxes[v]); scores[i] = max(scores[i], scores[v]); self.extent_swaps += 1
        # the matched views boxes stay in the pool; class-wise NMS then keeps one box per object


def build(environ=os.environ):
    # DRONE_EXTENT_CLASS_POLICY may be a JSON string or the path of a file holding one: JSON survives the ssh/docker
    # quoting layers far more reliably as a file.
    raw = environ.get('DRONE_EXTENT_CLASS_POLICY', '')
    if raw and os.path.isfile(raw):
        environ = dict(environ); environ['DRONE_EXTENT_CLASS_POLICY'] = open(raw).read().strip()
    install_class_extent_policy(environ)
    raw = environ.get('DRONE_ENS_MEMBERS', '[]')
    members = json.load(open(raw)) if os.path.isfile(raw) else json.loads(raw)  # a JSON list, or the path of a file holding one
    if not members:  # two-member shorthand
        tile, views = environ.get('DRONE_ENS_TILE_WEIGHTS'), environ.get('DRONE_ENS_VIEWS_WEIGHTS')
        if not tile or not views:
            raise ValueError('set DRONE_ENS_MEMBERS, or DRONE_ENS_TILE_WEIGHTS and DRONE_ENS_VIEWS_WEIGHTS')
        members = [dict(kind='tile', weights=tile, scales=[float(v) for v in environ.get('DRONE_ENS_TILE_SCALES', '4,2,1').split(',')]),
                   dict(kind='views', weights=views, imgsz=int(environ.get('DRONE_ENS_VIEWS_IMGSZ', '1280')))]
    return EnsembleDetector(members, device=environ.get('DRONE_DEVICE', 'cuda:0'), conf=float(environ.get('DRONE_SYNTH_CONF', '0.25')),
                            iou=float(environ.get('DRONE_SYNTH_IOU', '0.6')), half=environ.get('DRONE_SYNTH_HALF', '1') == '1',
                            max_det=int(environ.get('DRONE_SYNTH_MAXDET', '300')), class_conf=json.loads(environ.get('DRONE_SYNTH_CLASS_CONF', '{}')),
                            views_extent=environ.get('DRONE_ENS_VIEWS_EXTENT', '0') == '1',
                            tile_box_scale=json.loads(environ.get('DRONE_ENS_TILE_BOX_SCALE', '{}')))
