"""Compatibility forwarding for an already queued Runpod callback URL.

The controller and seed search run entirely on Hetzner. New submissions should
use Hetzner directly; this bridge only keeps an immutable old callback usable.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import requests

UPSTREAM = 'http://46.62.244.29:9064'
OLD_PREFIX = '/seed-live-eval-20260920'
NEW_PREFIX = '/seed-live-mode144'

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def proxy(self):
        path = self.path
        if path.startswith(OLD_PREFIX):
            path = NEW_PREFIX + path[len(OLD_PREFIX):]
        if path not in ('/health', NEW_PREFIX + '/predict', NEW_PREFIX + '/status'):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try:
            r = requests.request(self.command, UPSTREAM + path, data=body,
                                 headers={'Content-Type': 'application/json'}, timeout=10)
            self.send_response(r.status_code)
            self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(r.content)))
            self.end_headers()
            self.wfile.write(r.content)
        except requests.RequestException:
            self.send_error(502, 'Hetzner upstream unavailable')

    do_GET = proxy
    do_POST = proxy

    def log_message(self, fmt, *args):
        pass

if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 19123), Handler).serve_forever()
