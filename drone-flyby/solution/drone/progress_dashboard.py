"""Loopback-only, read-only live view of drone discovery artifacts."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/drone/mined'
IMAGES = ROOT / 'data/drone/reconstructed-validation'
DISCOVERY = ROOT / 'data/drone/discovery'

def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

@lru_cache(maxsize=32)
def preview(frame):
    from PIL import Image
    with Image.open(IMAGES / f'frame_{frame:06d}.png') as im:
        im.thumbnail((1920, 1080))
        bg = Image.new('RGB', im.size, '#182331')
        if im.mode == 'RGBA':
            bg.paste(im, mask=im.getchannel('A'))
        else:
            bg.paste(im)
        buf = BytesIO()
        bg.save(buf, 'JPEG', quality=88)
        return buf.getvalue()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, content, mime, code=200):
        try:
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # Moving the frame slider cancels the previous image request.
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            return self.send(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/progress':
            if (DISCOVERY / 'status.json').is_file():
                progress = read_json(DISCOVERY / 'status.json', {})
                try:
                    os.kill(int(progress.get('worker_pid',0)), 0)
                    running = bool(progress.get('worker_pid'))
                except PermissionError:
                    running = True
                except (OSError, ValueError, TypeError):
                    running = False
                progress['worker_alive'] = running
                progress['heartbeat_age_seconds'] = round(time.time()-progress.get('heartbeat_epoch',0),1)
                dataset = read_json(DISCOVERY / 'confirmed-seeds.json', {'annotations':[]})
                return self.send(json.dumps({'progress':progress,'dataset':dataset,
                    'server_time':datetime.now(timezone.utc).isoformat()}).encode(), 'application/json')
            dataset = read_json(DATA / 'object-presence-pass.json', {})
            progress = read_json(DATA / 'progress.json', {'status': 'unknown', 'message': 'Waiting for worker status', 'frames_reviewed': []})
            manifest = read_json(IMAGES / 'manifest.json', [])
            payload = {'progress': progress, 'dataset': dataset,
                       'images': [{k: row.get(k) for k in ('frame', 'native_coverage', 'complete')} for row in manifest],
                       'server_time': datetime.now(timezone.utc).isoformat(),
                       'dataset_updated_at': datetime.fromtimestamp((DATA / 'object-presence-pass.json').stat().st_mtime, timezone.utc).isoformat() if dataset else None}
            return self.send(json.dumps(payload).encode(), 'application/json')
        if path.startswith('/frame/'):
            try:
                frame = int(path.split('/')[-1])
                if not 1 <= frame <= 249:
                    raise ValueError()
                return self.send(preview(frame), 'image/jpeg')
            except (ValueError, OSError):
                return self.send(b'Frame unavailable', 'text/plain', 404)
        downloads = {'/download/json': DISCOVERY / 'confirmed-seeds.json',
                     '/download/csv': DISCOVERY / 'confirmed-seeds.csv'}
        if path in downloads and downloads[path].is_file():
            return self.send(downloads[path].read_bytes(), 'application/json' if path.endswith('json') else 'text/csv')
        return self.send(b'Not found', 'text/plain', 404)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9054)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Drone progress: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
