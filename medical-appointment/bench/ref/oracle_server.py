"""Local server for the human-oracle pass over the reference transcripts.

Serves bench/ref/oracle.html at http://localhost:9060/, the MP3s under /audio/,
and a small JSON API over the bench/ref/*.txt files, which are the checkpoint:

  GET  /api/list            every conversation with its status (checked / edited / untouched) and progress line
  GET  /api/text/<stem>     the file's lines
  POST /api/text/<stem>     body = full file text; written atomically; nothing else touches the file

Standard library only. Run from anywhere:

    python bench/ref/oracle_server.py            # then open http://localhost:9060/
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent            # bench/ref
CASE = HERE.parent.parent                          # medical-appointment
AUDIO = CASE / 'data' / 'audio'
PORT = int(os.environ.get('ORACLE_PORT', '9060'))
STEM = re.compile(r'^conversation_sample_\d+$')


def status_of(path: Path) -> dict:
    lines = path.read_text(encoding='utf-8').splitlines()
    checked = any(l.strip() == '# checked' for l in lines[:3])
    prog = next((int(m.group(1)) for l in lines[:3] for m in [re.match(r'#\s*progress:\s*(\d+)', l)] if m), 0)
    n = sum(1 for l in lines if l.startswith('['))
    tagged = sum(1 for l in lines if re.match(r'^\[[\d:.]+\]\s*\[[DP?]\]', l))
    return {'checked': checked, 'progress': prog, 'lines': n, 'tagged': tagged, 'edited': prog > 0 or checked}


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, fmt, *args):  # quieter
        if '/api/' in (args[0] if args else ''):
            return
        super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = unquote(self.path.split('?', 1)[0])
        if p == '/' or p == '/index.html':
            self.path = '/oracle.html'
            return super().do_GET()
        if p == '/api/list':
            files = sorted(HERE.glob('conversation_sample_*.txt'), key=lambda q: int(re.sub(r'\D', '', q.stem)))
            return self._json([{'stem': f.stem, **status_of(f)} for f in files])
        if p.startswith('/api/text/'):
            stem = p[len('/api/text/'):]
            if not STEM.match(stem) or not (HERE / f'{stem}.txt').exists():
                return self._json({'error': 'no such file'}, 404)
            return self._json({'stem': stem, 'text': (HERE / f'{stem}.txt').read_text(encoding='utf-8')})
        if p.startswith('/audio/'):
            return self._audio(p[len('/audio/'):])
        return super().do_GET()

    def do_POST(self):
        p = unquote(self.path)
        if not p.startswith('/api/text/'):
            return self._json({'error': 'not found'}, 404)
        stem = p[len('/api/text/'):]
        if not STEM.match(stem):
            return self._json({'error': 'bad name'}, 400)
        n = int(self.headers.get('Content-Length', '0'))
        text = self.rfile.read(n).decode('utf-8')
        if not text.strip():
            return self._json({'error': 'refusing to write an empty file'}, 400)
        target = HERE / f'{stem}.txt'
        tmp = target.with_suffix('.txt.tmp')
        tmp.write_text(text if text.endswith('\n') else text + '\n', encoding='utf-8', newline='\n')
        os.replace(tmp, target)
        return self._json({'ok': True, **status_of(target)})

    def _audio(self, name: str):
        f = AUDIO / name
        if not name.endswith('.mp3') or not f.exists():
            self.send_error(404); return
        size = f.stat().st_size
        rng = self.headers.get('Range')
        start, end = 0, size - 1
        if rng and rng.startswith('bytes='):
            a, _, b = rng[6:].partition('-')
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if (b and a) else size - 1
            end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header('Content-Type', 'audio/mpeg')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(length))
        if rng:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with open(f, 'rb') as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main():
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    print(f'oracle server on http://localhost:{PORT}/  (files: {HERE})', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
