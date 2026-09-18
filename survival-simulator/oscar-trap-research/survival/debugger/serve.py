"""Local Survival Lab with automatic discovery of completed workspace replays."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import json
import mimetypes
from catalog import ReplayCatalog


class ReplayHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, catalog, **kwargs):
        self.catalog = catalog
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # Existing browser tabs must see updated controls and catalog entries.
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        self._serve(False)

    def do_HEAD(self):
        self._serve(True)

    def _serve(self, head):
        route = urlsplit(self.path).path.lstrip('/')
        if route == 'recordings/manifest.json':
            body = json.dumps(self.catalog.snapshot(), separators=(',', ':')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)
            return
        if route.startswith('replays/'):
            path = self.catalog.resolve(route)
            if path is None:
                self.send_error(404, 'Replay unavailable or changed; refresh the recordings list')
                return
            try:
                with path.open('rb') as source:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/gzip' if path.suffix == '.gz' else 'application/json')
                    self.send_header('Content-Length', str(path.stat().st_size))
                    self.end_headers()
                    if not head:
                        self.copyfile(source, self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if head:
            super().do_HEAD()
        else:
            super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9053)
    parser.add_argument('--open', action='store_true', help='Open the viewer in your default browser')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    catalog = ReplayCatalog()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(ReplayHandler, directory=str(root), catalog=catalog))
    catalog.start()
    print(f'Survival debugger: http://127.0.0.1:{args.port} (automatic replay discovery)', flush=True)
    if args.open:
        import threading, webbrowser
        threading.Thread(target=webbrowser.open, args=(f'http://127.0.0.1:{args.port}',), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        catalog.stop_event.set()
        server.server_close()
