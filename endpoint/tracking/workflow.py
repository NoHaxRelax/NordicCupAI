"""Workflow-facing frame handler, legal camera sweep and observed motion clock.

No network, detector weights, annotations or benchmark data are loaded here.
Pass detections from your own detector; process returns the organizer response.
One workflow instance handles one sequence, serially.
"""
import base64
from dataclasses import asdict

import numpy as np

from .motion import CalibrationError, MotionModel, ViewGeometry, calibrate_images, finite
from .revisit import Detection, RevisitConfig, RevisitTracker, frame_rows


class LevelOneSweep:
    """Upper sweep with center crops or full overviews between the two sides.

    mode 'l1': L1 left, centre (or L0 overview), right, centre. mode 'l2_top':
    native L2 crops bouncing along the upper row; hops stay within the L2
    move limit, so every column is delivered at native resolution about
    every six frames. Level changes step one level at a time.
    """
    def __init__(self, vertical_fraction=0., *, overview_between_sides=False, mode='l1'):
        if not 0 <= vertical_fraction <= 1:
            raise ValueError('Vertical fraction must be in [0,1]')
        if mode not in ('l1', 'l2_top'):
            raise ValueError("Camera mode must be 'l1' or 'l2_top'")
        self.vertical_fraction = vertical_fraction
        self.overview_between_sides = overview_between_sides
        self.mode = mode
        self.waypoint = 0

    def l2_waypoints(self, request):
        width, height = request['original_width'], request['original_height']
        limit = 550.  # inside the L2 move limit of half its view diagonal (551 px)
        low, high = 480., width-480.
        count = int(np.ceil((high-low)/limit))+1
        xs = np.linspace(low, high, count)
        y = 270.+self.vertical_fraction*(height-540.)
        order = list(range(count))+list(range(count-2, 0, -1))  # bounce
        return xs, y, order

    def next_view(self, request, *, focus_box=None, overview=False):
        constraints = request['camera_constraints']; view = request['view']
        bounds = {b['resolution_level']: b for b in constraints['center_bounds']
                  if b['resolution_level'] in constraints['allowed_resolution_levels']}
        current = np.array([view['center_x'], view['center_y']], float)
        if overview and 0 in bounds:
            level, target = 0, np.array([request['original_width']/2, request['original_height']/2])
        elif focus_box is not None and 2 in bounds:
            from .motion import box_array
            box = box_array(focus_box); level, target = 2, (box[:2]+box[2:])/2
        elif self.mode == 'l2_top' and (2 in bounds or 1 in bounds):
            xs, y, order = self.l2_waypoints(request)
            level, target = 2, np.array([xs[order[self.waypoint % len(order)]], y])
            if view['resolution_level'] == 2 and np.linalg.norm(current-target) < 1:
                self.waypoint = (self.waypoint+1) % len(order)
                target = np.array([xs[order[self.waypoint]], y])
            if level not in bounds:
                level = 1  # one level at a time: approach through L1
        elif 1 in bounds:
            level = 1; b = bounds[level]
            xs = [b['minimum_center_x'], (b['minimum_center_x']+b['maximum_center_x'])/2,
                  b['maximum_center_x'], (b['minimum_center_x']+b['maximum_center_x'])/2]
            y = b['minimum_center_y']+self.vertical_fraction*(b['maximum_center_y']-b['minimum_center_y'])
            def destination(waypoint):
                if self.overview_between_sides and waypoint % 2:
                    return 0, np.array([request['original_width']/2, request['original_height']/2])
                return 1, np.array([xs[waypoint], y])
            level, target = destination(self.waypoint)
            if view['resolution_level'] == level and np.linalg.norm(current-target) < 1:
                self.waypoint = (self.waypoint+1) % 4
                level, target = destination(self.waypoint)
            if level not in bounds:
                # A level-2 detour must return through level 1 before overview.
                level = 1
        else:
            return None
        b = bounds[level]
        low = np.array([b['minimum_center_x'], b['minimum_center_y']], float)
        high = np.array([b['maximum_center_x'], b['maximum_center_y']], float)
        target = np.clip(target, low, high)
        exempt = level == 0 and constraints.get('full_view_reset_exempt_from_delta', False)
        limit = float(constraints['maximum_center_delta'])
        if not exempt and np.linalg.norm(target-current) > limit:
            # Find a feasible centre in the target level, then advance towards
            # the goal while remaining inside its bounds and movement circle.
            nearest = np.clip(current, low, high)
            if np.linalg.norm(nearest-current) > max(0., limit-1):
                return None
            lo, hi = 0., 1.
            for _ in range(35):
                fraction = (lo+hi)/2
                point = nearest+fraction*(target-nearest)
                if np.linalg.norm(point-current) <= max(0., limit-1):
                    lo = fraction
                else:
                    hi = fraction
            target = nearest+lo*(target-nearest)
        target = np.rint(target).astype(int)
        if not exempt and np.linalg.norm(target-current) > limit+1e-9:
            return None
        return {'resolution_level': int(level), 'center_x': int(target[0]), 'center_y': int(target[1])}


def decode_image(request):
    import cv2
    encoded = request['view'].get('image')
    if not encoded:
        raise ValueError('Supply an image array or request.view.image for calibration')
    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Invalid PNG image')
    return image


def observed_step(previous, previous_view, current, current_view, model, tick, fallback):
    """Recognize zero/normal/double steps from overlap in delivered pixels.

    This coarse constant-speed clock is not a general speed estimator. If
    overlapping background evidence is insufficient, report the frame-gap
    fallback explicitly. No object labels or future images are consumed.
    """
    import cv2
    from aligned_views import patches as aligned_patches
    patches, region, scale, size = aligned_patches(previous, current, previous_view, current_view)
    info = {'mode': 'frame_gap_fallback', 'motion_step': float(fallback)}
    if patches is None:
        return float(fallback), {**info, 'reason': 'insufficient_overlap'}
    points = cv2.goodFeaturesToTrack(patches[0], 300, .025, 10, blockSize=7)
    if points is None or len(points) < 12:
        return float(fallback), {**info, 'reason': 'insufficient_texture'}
    params = dict(winSize=(31, 31), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
    nxt, ok, err = cv2.calcOpticalFlowPyrLK(*patches, points, None, **params)
    if nxt is None:
        return float(fallback), {**info, 'reason': 'flow_failed'}
    back, okback, _ = cv2.calcOpticalFlowPyrLK(patches[1], patches[0], nxt, None, **params)
    if back is None:
        return float(fallback), {**info, 'reason': 'reverse_flow_failed'}
    xy, end = points[:, 0], nxt[:, 0]
    keep = (ok[:, 0] == 1) & (okback[:, 0] == 1) & (err[:, 0] < 18)
    keep &= np.linalg.norm(back[:, 0]-xy, axis=1) < 1
    keep &= np.all(end > 12, axis=1) & np.all(end < size-12, axis=1)
    if keep.sum() < 12:
        return float(fallback), {**info, 'reason': 'insufficient_consistent_flow'}
    source_points = (xy[keep]+.5)*scale-.5+region[:2]
    expected = model.points(source_points, tick, tick+1)-source_points
    actual = (end[keep]-xy[keep])*scale
    valid = np.linalg.norm(expected, axis=1) > 2
    if valid.sum() < 12:
        return float(fallback), {**info, 'reason': 'weak_expected_motion'}
    ratios = np.sum(actual[valid]*expected[valid], axis=1)/np.sum(expected[valid]**2, axis=1)
    ratio = float(np.median(ratios)); spread = float(np.median(np.abs(ratios-ratio)))
    step = int(round(ratio))
    if spread > .2 or abs(ratio-step) > .25 or step < 0 or step > max(2, 2*fallback):
        return float(fallback), {**info, 'reason': 'unsupported_speed_or_inconsistent_flow', 'ratio': ratio, 'spread': spread}
    return float(step), {'mode': 'observed_background', 'motion_step': float(step),
                         'ratio': ratio, 'spread': spread, 'features': int(valid.sum())}


class DroneTrackingWorkflow:
    """A serial process(request, detections) interface for the real pipeline.

    Calibration uses two delivered images from this sequence. Warm-up replies
    contain current detections and hold the overview. Reliable repeat
    detections refresh boxes automatically; crop absence never removes a track.
    No class-conditioned shape model is enabled.
    """
    def __init__(self, config=None, *, observe_motion=True, vertical_fraction=0., overview_between_sides=False, camera_mode='l1',
                 revisit_every=0, revisit_min_age=6.):
        self.config = config or RevisitConfig()
        self.prior = self.config.load_prior()
        self.observe_motion = observe_motion
        self.camera = LevelOneSweep(vertical_fraction, overview_between_sides=overview_between_sides, mode=camera_mode)
        # Optional native revisits: every k-th tracking frame, aim the camera at
        # the reachable track that has gone longest without a fresh observation.
        if revisit_every < 0 or revisit_min_age < 0:
            raise ValueError('Revisit settings must be nonnegative')
        self.revisit_every = int(revisit_every)
        self.revisit_min_age = float(revisit_min_age)
        self.sequence_id = None
        self.tracker = None
        self.warmup = None
        self.previous = None
        self.last_frame = None
        self.motion_tick = None
        self.last_request_id = None
        self.last_response = None
        self.diagnostics = {}
        self.inspection_attempts = {}

    def process(self, request, detections, *, image=None, tick=None, detector_ran=True, focus_box=None):
        import copy
        view = ViewGeometry.from_request(request)
        if self.sequence_id is not None and request['sequence_id'] != self.sequence_id:
            raise ValueError('New sequence: instantiate a separate DroneTrackingWorkflow')
        if self.last_request_id == request['request_id']:
            if self.last_frame != request['frame_index']:
                raise ValueError('Request ID reused at another frame')
            return copy.deepcopy(self.last_response)
        frame = int(request['frame_index'])
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError('Received an out-of-order frame')
        detections = [d if isinstance(d, Detection) else Detection(**d) for d in detections]
        for d in detections:
            view.box_to_source(d.box)
        if not detector_ran and detections:
            raise ValueError('Detector skipped but detections supplied')
        if image is None and (self.tracker is None or self.observe_motion):
            image = decode_image(request)
        if image is not None and (image.dtype != np.uint8 or image.shape[:2] != view.image_size[::-1]):
            raise ValueError('Image must be uint8 and match delivered view dimensions')
        motion = float(finite(tick, 'motion tick')) if tick is not None else float(frame)
        timing = {'mode': 'supplied_clock' if tick is not None else 'frame_index'}
        if self.tracker is not None and tick is None:
            step = frame-self.last_frame
            if self.observe_motion and self.previous is not None:
                old, old_view = self.previous
                step, timing = observed_step(old, old_view, image, view, self.tracker.model, self.motion_tick, step)
            else:
                timing = {'mode': 'frame_gap_fallback', 'reason': 'image_clock_disabled_or_resumed'}
            motion = self.motion_tick+step
        if self.motion_tick is not None and motion < self.motion_tick:
            raise ValueError('Motion clock moved backwards')
        self.sequence_id = request['sequence_id']
        calibration_error = None
        if self.tracker is None and self.warmup is not None:
            old_image, old_view, old_tick, old_frame, old_detections, old_ran = self.warmup
            try:
                model = calibrate_images(old_image, image, old_view, view, first_tick=old_tick, second_tick=motion)
                self.tracker = RevisitTracker(model, self.sequence_id, self.config)
                self.tracker.update(old_detections, old_view, old_tick, old_frame, detector_ran=old_ran)
            except CalibrationError as exc:
                calibration_error = str(exc)
        if self.tracker is None:
            self.warmup = (image.copy(), view, motion, frame, detections, detector_ran)
            rows, counts = [], {}
            for row in frame_rows(detections, view, self.config, self.prior):
                if counts.get(row['object_id'], 0) < 100 and len(rows) < 500:
                    rows.append({'object_id': row['object_id'], 'confidence': row['confidence'],
                                 'bbox': (np.array(row['bbox_source_xyxy'])/np.tile(view.source_size, 2)).tolist()})
                    counts[row['object_id']] = counts.get(row['object_id'], 0)+1
            response = {'request_id': request['request_id'], 'frame': request['frame'], 'annotations': rows[:500],
                        'requested_view': self.camera.next_view(request, overview=True)}
            events = []
        else:
            self.warmup = None
            self.tracker.update(detections, view, motion, frame, detector_ran=detector_ran)
            if focus_box is None and self.revisit_every and frame % self.revisit_every == 0:
                focus_box = self.uncertain_small_track(request, motion)
            response = self.tracker.response(request, tick=motion,
                requested_view=self.camera.next_view(request, focus_box=focus_box))
            events = self.tracker.events
        self.motion_tick = motion; self.last_frame = frame; self.last_request_id = request['request_id']
        self.previous = (image.copy(), view) if image is not None else None
        self.last_response = copy.deepcopy(response)
        self.diagnostics = {'status': 'tracking' if self.tracker else 'calibrating', 'motion_tick': motion,
                            'timing': timing, 'calibration_error': calibration_error, 'events': events,
                            'camera_feedback': request.get('camera_command_feedback'),
                            'tracks': self.tracker.predictions(motion) if self.tracker else [],
                            'pre_update_predictions': self.tracker.pre_update_predictions if self.tracker else [],
                            'focus_box': focus_box}
        return response

    def uncertain_small_track(self, request, tick):
        """Inspect a reachable uncertain small object without withholding output."""
        if request['view']['resolution_level'] != 1:
            return None
        constraints=request['camera_constraints']
        if 2 not in constraints['allowed_resolution_levels']:
            return None
        bounds=next(b for b in constraints['center_bounds'] if b['resolution_level']==2)
        current=np.array([request['view']['center_x'],request['view']['center_y']],float)
        frame=request['frame_index']; candidates=[]
        for row in self.tracker.predictions(tick):
            identity=row['track_id']
            if row['object_id'] not in ('small_launcher','medium_launcher','ta-ta') or row['confidence']>=.6 or row.get('provisional'):
                continue
            if frame-self.inspection_attempts.get(identity,-1000000)<24:
                continue
            box=self.tracker._box(self.tracker.tracks[identity],tick+1)
            center=(box[:2]+box[2:])/2
            target=np.clip(center,[bounds['minimum_center_x'],bounds['minimum_center_y']],
                                  [bounds['maximum_center_x'],bounds['maximum_center_y']])
            if np.linalg.norm(target-current)>constraints['maximum_center_delta']-2:
                continue
            if np.any(box[:2]<target-[478,268]) or np.any(box[2:]>target+[478,268]):
                continue
            candidates.append((center[1]/request['original_height'],identity,box.tolist()))
        if not candidates:return None
        _,identity,box=max(candidates)
        self.inspection_attempts[identity]=frame
        self.tracker.events.append({'event':'l2_inspection_requested','track_id':identity})
        return box

    def stale_track(self, request, tick):
        """Box of the oldest-anchored track a one-step L2 move can reach, or None."""
        constraints = request['camera_constraints']; view = request['view']
        if 2 not in constraints['allowed_resolution_levels']:
            return None
        bounds = next((b for b in constraints['center_bounds'] if b['resolution_level'] == 2), None)
        if bounds is None:
            return None
        current = np.array([view['center_x'], view['center_y']], float)
        limit = float(constraints['maximum_center_delta'])
        best = None
        for row in self.tracker.predictions(tick):
            age = tick-row['anchor_tick']
            if row.get('provisional') or age < self.revisit_min_age:
                continue
            box = np.array(row['bbox_source_xyxy']); centre = (box[:2]+box[2:])/2
            centre = np.clip(centre, [bounds['minimum_center_x'], bounds['minimum_center_y']],
                             [bounds['maximum_center_x'], bounds['maximum_center_y']])
            if np.linalg.norm(centre-current) > limit:
                continue
            if best is None or age > best[0]:
                best = (age, box.tolist())
        return best[1] if best else None

    def to_dict(self):
        if self.tracker is None:
            raise ValueError('Save the workflow after calibration completes')
        return {'version': 1, 'tracker': self.tracker.to_dict(), 'config': asdict(self.config),
                'observe_motion': self.observe_motion, 'camera_waypoint': self.camera.waypoint,
                'overview_between_sides': self.camera.overview_between_sides, 'camera_mode': self.camera.mode,
                'revisit_every': self.revisit_every, 'revisit_min_age': self.revisit_min_age,
                'vertical_fraction': self.camera.vertical_fraction, 'last_frame': self.last_frame,
                'motion_tick': self.motion_tick, 'last_request_id': self.last_request_id,
                'last_response': self.last_response, 'inspection_attempts': self.inspection_attempts}

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported workflow state version')
        result = cls(RevisitConfig(**data['config']), observe_motion=data['observe_motion'],
                     vertical_fraction=data['vertical_fraction'],
                     overview_between_sides=data.get('overview_between_sides', False),
                     camera_mode=data.get('camera_mode', 'l1'),
                     revisit_every=data.get('revisit_every', 0), revisit_min_age=data.get('revisit_min_age', 6.))
        result.tracker = RevisitTracker.from_dict(data['tracker']); result.sequence_id = result.tracker.sequence_id
        result.camera.waypoint = data['camera_waypoint']; result.last_frame = data['last_frame']
        result.motion_tick = data['motion_tick']; result.last_request_id = data['last_request_id']
        result.inspection_attempts = data.get('inspection_attempts', {})
        result.last_response = data['last_response']
        return result
