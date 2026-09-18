"""Run the tracking workflow on JSONL requests or serve a local YOLO callback.

No weights are downloaded. --weights must point to a locally available,
fine-tuned checkpoint whose class names match the competition classes.
Without --weights, JSONL records must contain an explicit detections list.
"""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .revisit import Detection, RevisitConfig
from .tracker import OBJECT_CLASSES
from .workflow import DroneTrackingWorkflow, decode_image


class UltralyticsDetector:
    def __init__(self, weights, *, device='cpu', image_size=960):
        path = Path(weights)
        if not path.is_file():
            raise ValueError('Weights must be an existing local file; no automatic download')
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('The YOLO adapter needs torch and ultralytics in the detector environment') from exc
        with redirect_stdout(sys.stderr):
            self.model = YOLO(str(path))
        if len(self.model.names) != len(OBJECT_CLASSES) or set(self.model.names.values()) != OBJECT_CLASSES:
            raise ValueError('Checkpoint must have exactly the 16 competition class names')
        self.device, self.image_size = device, image_size

    def __call__(self, image):
        import cv2
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        with redirect_stdout(sys.stderr):
            result = self.model.predict(image, imgsz=self.image_size, device=self.device,
                                        conf=.4, verbose=False)[0]
        rows = result.boxes.data.cpu().tolist()
        return [Detection(self.model.names[int(row[5])], tuple(row[:4]), float(row[4])) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='JSONL file; omit to read stdin')
    parser.add_argument('--output', type=Path, help='JSONL responses; omit for stdout')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--state-in', type=Path)
    parser.add_argument('--state-out', type=Path)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--frame-clock-only', action='store_true')
    parser.add_argument('--adapt-edges', action='store_true', help='Experimental per-track adaptation; off by default')
    parser.add_argument('--overview-between-sides', action='store_true', help='L1 left, L0 overview, L1 right, L0 overview')
    parser.add_argument('--serve', action='store_true', help='Local /predict HTTP endpoint; requires --weights')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.serve and args.weights is None:
        parser.error('--serve requires local detector --weights')
    if args.serve and (args.input or args.output):
        parser.error('--input/--output apply to JSONL mode only')
    detector = UltralyticsDetector(args.weights, device=args.device) if args.weights else None
    workflow = (DroneTrackingWorkflow.from_dict(json.loads(args.state_in.read_text())) if args.state_in else
                DroneTrackingWorkflow(RevisitConfig(adapt_edges=args.adapt_edges), observe_motion=not args.frame_clock_only,
                                      overview_between_sides=args.overview_between_sides))
    diagnostic_file = args.diagnostics.open('a') if args.diagnostics else None

    def handle(record):
        nonlocal workflow
        request = record.get('request', record)
        # Each sequence gets independent camera, association and motion state.
        if workflow.sequence_id is not None and workflow.sequence_id != request['sequence_id']:
            workflow = DroneTrackingWorkflow(workflow.config, observe_motion=workflow.observe_motion,
                                             vertical_fraction=workflow.camera.vertical_fraction,
                                             overview_between_sides=workflow.camera.overview_between_sides)
        image = decode_image(request)
        if detector is not None:
            detections = detector(image)
        elif 'detections' in record:
            detections = record['detections']
        else:
            raise ValueError('JSONL records need detections when no detector weights are supplied')
        answer = workflow.process(request, detections, image=image, tick=record.get('motion_tick'),
                                  detector_ran=record.get('detector_ran', True), focus_box=record.get('focus_box'))
        if diagnostic_file:
            diagnostic_file.write(json.dumps({'request_id': request['request_id'], **workflow.diagnostics}, allow_nan=False)+'\n')
            diagnostic_file.flush()
        if args.state_out and workflow.tracker is not None:
            temporary = args.state_out.with_name(args.state_out.name+'.tmp')
            temporary.write_text(json.dumps(workflow.to_dict(), allow_nan=False)+'\n')
            temporary.replace(args.state_out)
        return answer

    if args.serve:
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != '/predict':
                    return self.send_error(404)
                if not lock.acquire(blocking=False):
                    return self.send_error(503, 'Previous frame is still being processed')
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 8*1024*1024:
                        return self.send_error(413)
                    request = json.loads(self.rfile.read(length))
                    body = json.dumps(handle(request), allow_nan=False).encode()
                    self.send_response(200); self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
                except (ValueError, KeyError, TypeError) as exc:
                    print(f'Frame rejected: {type(exc).__name__}: {exc}', file=sys.stderr)
                    self.send_error(400, 'Invalid or out-of-order frame; inspect local diagnostics')
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    print(f'Frame failed: {type(exc).__name__}: {exc}', file=sys.stderr)
                    self.send_error(500, 'Inference failed; inspect local diagnostics')
                finally:
                    lock.release()
            def log_message(self, *_):
                pass
        print(f'Local tracking callback on {args.host}:{args.port}/predict', file=sys.stderr)
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    else:
        source = args.input.open() if args.input else sys.stdin
        destination = args.output.open('w') if args.output else sys.stdout
        try:
            for line in source:
                if line.strip():
                    destination.write(json.dumps(handle(json.loads(line)), allow_nan=False)+'\n'); destination.flush()
        finally:
            if args.input: source.close()
            if args.output: destination.close()
            if diagnostic_file: diagnostic_file.close()


if __name__ == '__main__':
    main()
