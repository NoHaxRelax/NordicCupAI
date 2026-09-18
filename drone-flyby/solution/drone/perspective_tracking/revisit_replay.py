"""Legal-camera reference replay of the plug-in workflow using supplied boxes.

Scores unseen-between-revisit forecasts separately from current detections.
This is an integration/geometry test, not an actual detector or mAP benchmark.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from . import (Detection, DeterministicTracker, DroneTrackingWorkflow,
               RevisitConfig, ViewGeometry)
from .revisit import overlap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'artifacts/drone-source-2026-09-17'))
from local_evaluator import Camera
from dtos import DroneFlybyPredictResponseDto
from utils import validate_response


def run(output, *, adapt_edges=False, detour=False, save_inputs=False, overview_between_sides=False):
    output.mkdir(parents=True, exist_ok=True)
    workflow = DroneTrackingWorkflow(RevisitConfig(adapt_edges=adapt_edges), overview_between_sides=overview_between_sides)
    camera = Camera(); baseline = None; anchors = {}; rows = []; traces = []; timings = []
    unseen = []; unmatched = []; responses = []; inputs = []; detection_sizes = []
    for frame in range(25):
        source = cv2.imread(str(ROOT/f'data/drone/reference/helsinki/images/frame_{frame:06d}.png'))
        truth = {a['object_id']: np.array(a['bbox'], float) for a in json.loads(
            (ROOT/f'data/drone/reference/helsinki/annotations/frame_{frame:06d}.json').read_text())['annotations']}
        region = np.array(camera.source_region); x1, y1, x2, y2 = region
        image = cv2.resize(source[y1:y2, x1:x2], (960, 540), interpolation=cv2.INTER_AREA)
        req = {'sequence_id': 'reference-revisit', 'request_id': f'replay-{frame}', 'frame': frame,
               'frame_index': frame, 'original_width': 3840, 'original_height': 2160,
               'view': {'resolution_level': camera.resolution_level, 'center_x': camera.center_x,
                        'center_y': camera.center_y, 'width': 960, 'height': 540,
                        'source_region_xyxy': list(map(int, region))}, 'camera_constraints': camera.constraints()}
        view = ViewGeometry.from_request(req); detections = []
        for label, box in truth.items():
            visible = np.r_[np.maximum(box[:2], region[:2]), np.minimum(box[2:], region[2:])]
            if np.all(visible[2:] > visible[:2]):
                complete = bool(np.all(box[:2] > 0) and np.all(box[2:] < [3839, 2159]) and
                                np.all(box[:2] > region[:2]) and np.all(box[2:] < region[2:]))
                detections.append(Detection(label, tuple(view.box_from_source(visible)), .99, complete))
                local = np.array(detections[-1].box)
                detection_sizes.append({'frame': frame, 'class': label, 'level': camera.resolution_level,
                                        'complete': complete, 'width': float(local[2]-local[0]),
                                        'height': float(local[3]-local[1])})
        focus = [3300, 500, 3360, 570] if detour and frame in (8, 9, 10) else None
        start = time.perf_counter()
        answer = workflow.process(req, detections, image=image, focus_box=focus)
        timings.append((time.perf_counter()-start)*1000)
        validate_response(DroneFlybyPredictResponseDto.model_validate(answer))
        responses.append(answer)
        if save_inputs:
            _, png = cv2.imencode('.png', image)
            req['view']['image'] = base64.b64encode(png).decode()
            inputs.append({'request': req, 'detections': [{'label': d.label, 'box': list(d.box),
                'confidence': d.confidence, 'complete': d.complete} for d in detections], 'focus_box': focus})
        if workflow.tracker:
            if baseline is None:
                baseline = DeterministicTracker(workflow.tracker.model, req['sequence_id'])
            for track in workflow.tracker.tracks.values():
                if track.label not in anchors:
                    tick, box = track.history[0]
                    baseline.add(track.label, track.label, box, tick, confidence=.99)
                    anchors[track.label] = tick
            predictions = workflow.tracker.predictions(workflow.motion_tick)
            grouped = {}
            for prediction in predictions:
                grouped.setdefault(prediction['object_id'], []).append(prediction)
            for label, box in truth.items():
                if label not in anchors:
                    unseen.append({'frame': frame, 'class': label})
                    continue
                if workflow.motion_tick <= anchors[label]:
                    continue
                current = grouped.get(label, [])
                best = max(current, key=lambda r: overlap(np.array(r['bbox_source_xyxy']), box), default=None)
                refreshed = best is not None and best['anchor_tick'] == workflow.motion_tick
                blind = baseline.predict(label, workflow.motion_tick)
                for method, pred in [('revisit', best), ('one_observation', blind)]:
                    pred_box = np.array(pred['bbox_source_xyxy']) if pred else None
                    rows.append({'frame': frame, 'class': label, 'method': method, 'refreshed_now': refreshed,
                                 'iou': overlap(pred_box, box) if pred_box is not None else 0.,
                                 'prediction': pred_box.tolist() if pred_box is not None else None,
                                 'truth': box.tolist()})
            for label, current in grouped.items():
                extra = len(current)-(1 if label in truth else 0)
                if extra > 0: unmatched.append({'frame': frame, 'class': label, 'extra_predictions': extra})
        traces.append({'frame': frame, 'region': region.tolist(), 'level': camera.resolution_level,
                       'requested_view': answer['requested_view'], 'diagnostics': workflow.diagnostics})
        if answer['requested_view']:
            camera.apply(**answer['requested_view'])  # Any illegal command fails the replay.
    def summarize(rr):
        labels = sorted({r['class'] for r in rr})
        return {'boxes': len(rr), 'passing_iou50': sum(r['iou'] >= .5 for r in rr),
                'mean_iou': float(np.mean([r['iou'] for r in rr])) if rr else None,
                'classes': len(labels), 'whole_tested_class_tracks': sum(all(r['iou'] >= .5 for r in rr if r['class'] == label) for label in labels)}
    result = {'metadata': {'source': '25-frame official reference, ideal detections from visible official boxes',
                           'no_detector_accuracy_claim': True, 'no_added_noise': True,
                           'same_acquisition_and_future_frames': True, 'adapt_edges': adapt_edges,
                           'level_two_excursion': detour, 'camera': 'Actual organizer Camera applies each command',
                           'overview_between_sides': overview_between_sides,
                           'forecast_only': 'Current frame supplies no complete accepted box for this track.',
                           'coverage': ('Whole-frame overview every second frame after warm-up; each upper L1 side every fourth.'
                                        if overview_between_sides else 'Upper L1 sweep observes only its vertical band; no every-other-frame guarantee.')},
              'measurements': {method: {'all_after_acquisition': summarize([r for r in rows if r['method'] == method]),
                        'forecast_only': summarize([r for r in rows if r['method'] == method and not r['refreshed_now']])}
                               for method in ('one_observation', 'revisit')},
              'acquired_classes': sorted(anchors), 'unacquired_labelled_appearances': unseen,
              'unmatched_predictions': unmatched, 'response_count': len(responses),
              'runtime_ms': {'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95)),
                             'maximum': float(max(timings)), 'scope': 'Workflow including calibration/clock/tracking/planner; excludes detector, HTTP, PNG decode and source rendering.'}}
    (output/'measurements.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, data in [('predictions', rows), ('traces', traces), ('responses', responses), ('detection-sizes', detection_sizes)]:
        (output/f'{name}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in data))
    (output/'workflow-state.json').write_text(json.dumps(workflow.to_dict(), indent=2, allow_nan=False)+'\n')
    if save_inputs:
        (output/'requests-with-detections.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in inputs))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/drone-revisit-workflow/default')
    parser.add_argument('--adapt-edges', action='store_true')
    parser.add_argument('--level-two-detour', action='store_true')
    parser.add_argument('--save-inputs', action='store_true')
    parser.add_argument('--overview-between-sides', action='store_true')
    args = parser.parse_args()
    run(args.output, adapt_edges=args.adapt_edges, detour=args.level_two_detour, save_inputs=args.save_inputs,
        overview_between_sides=args.overview_between_sides)


if __name__ == '__main__':
    main()
