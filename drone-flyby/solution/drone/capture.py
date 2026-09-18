#!/usr/bin/env python3
"""Record authorized validation crops. No dataset download or evaluation routes."""
import argparse
import base64
import hashlib
import json
import math
import re
from pathlib import Path
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def next_view(request, target):
    """Approach a fixed native-resolution crop using supplied camera constraints."""
    view = request['view']
    constraints = request['camera_constraints']
    level = min(2, view['resolution_level'] + 1)
    if level not in constraints['allowed_resolution_levels']:
        return None
    bounds = next(b for b in constraints['center_bounds'] if b['resolution_level'] == level)
    x = max(bounds['minimum_center_x'], min(target[0], bounds['maximum_center_x']))
    y = max(bounds['minimum_center_y'], min(target[1], bounds['maximum_center_y']))
    dx, dy = x - view['center_x'], y - view['center_y']
    distance = math.hypot(dx, dy)
    limit = constraints['maximum_center_delta']
    if distance > limit:
        scale = max(0, limit - 2) / distance
        x = round(view['center_x'] + dx * scale)
        y = round(view['center_y'] + dy * scale)
    if not (bounds['minimum_center_x'] <= x <= bounds['maximum_center_x'] and
            bounds['minimum_center_y'] <= y <= bounds['maximum_center_y']):
        return None
    return {'resolution_level': level, 'center_x': x, 'center_y': y}


def save_request(request, output, target, annotations=None, probe_name=None):
    started = time.monotonic()
    image = base64.b64decode(request['view']['image'], validate=True)
    if image[:8] != b'\x89PNG\r\n\x1a\n' or len(image) < 24:
        raise ValueError('Expected PNG')
    size = struct.unpack('>II', image[16:24])
    if size != (request['view']['width'], request['view']['height']):
        raise ValueError('PNG and request dimensions disagree')
    sequence = hashlib.sha256(request['sequence_id'].encode()).hexdigest()[:16]
    folder = output / sequence
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{int(request['frame_index']):06d}_{hashlib.sha256(request['request_id'].encode()).hexdigest()[:12]}"
    (folder / f'{stem}.png').write_bytes(image)
    response = {'request_id': request['request_id'], 'frame': request['frame'],
                'annotations': annotations or [], 'requested_view': next_view(request, target)}
    metadata = {**request, 'view': {k: v for k, v in request['view'].items() if k != 'image'}}
    metadata.update(image_file=f'{stem}.png', image_sha256=hashlib.sha256(image).hexdigest(),
                    response=response, target_center=list(target), received_at=time.time(),
                    capture_ms=round((time.monotonic() - started) * 1000, 3))
    if probe_name:
        metadata['probe_name'] = probe_name
    (folder / f'{stem}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9053)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route-file', type=Path, required=True)
    parser.add_argument('--target', type=int, nargs=2, default=(480, 270))
    parser.add_argument('--plan-file', type=Path, help='Local JSON plan, held unchanged throughout each attempt')
    args = parser.parse_args()
    route = '/' + args.route_file.read_text().strip() + '/predict'
    args.output.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass  # Never log the private route or request headers.

        def do_GET(self):
            self.send_error(404)

        def do_POST(self):
            if self.path != route:
                self.send_error(404)
                return
            try:
                self.connection.settimeout(10)
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8 * 1024 * 1024:
                    self.send_error(413)
                    return
                request = json.loads(self.rfile.read(length))
                plan = json.loads(args.plan_file.read_text()) if args.plan_file else {}
                name = plan.get('name')
                if name and not re.fullmatch(r'[a-z0-9_-]{1,80}', name):
                    raise ValueError('Invalid probe name')
                annotations = plan.get('predictions_by_frame', {}).get(str(request['frame']), [])
                if len(annotations) > 500:
                    raise ValueError('Too many predictions')
                for annotation in annotations:
                    x1, y1, x2, y2 = annotation['bbox']
                    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and 0 <= annotation['confidence'] <= 1):
                        raise ValueError('Invalid prediction geometry')
                response = save_request(request, args.output / name if name else args.output,
                                        tuple(plan.get('target', args.target)), annotations, name)
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                print(f'Capture failed: {type(exc).__name__}', flush=True)
                self.send_error(400, 'Invalid capture request')

    print(f'Listening on loopback port {args.port}; target {args.target}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
