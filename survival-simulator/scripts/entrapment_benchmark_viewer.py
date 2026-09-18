"""Browse benchmark scores and replay recorded creature states with native draw()."""
import argparse
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlparse, parse_qs

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pygame
from src.elements.creature import Creature
from entrapment_benchmark import aggregate, verify_sources


class Replay:
    def __init__(self, folder):
        self.static = json.loads((folder/'static.json').read_text())
        with gzip.open(folder/'trajectory.jsonl.gz', 'rt') as stream:
            self.rows = [json.loads(line) for line in stream]
        self.catalog = {}
        for row in self.rows:
            self.catalog.update(row['births'])
        self.edges = []
        w, h = self.static['width'], self.static['height']
        for x, y, dx, dy in [*self.static['obstacles'], (0, 0, w, h)]:
            corners = [(x,y), (x+dx,y), (x+dx,y+dy), (x,y+dy)]
            self.edges.extend(zip(corners, corners[1:]+corners[:1]))
        self.world = pygame.Surface((w, h))
        self.vision = pygame.Surface((w, h), pygame.SRCALPHA)
        self.font = pygame.font.SysFont('monospace', 12)

    def row(self, tick):
        if not 0 <= tick < len(self.rows): raise IndexError(tick)
        return self.rows[tick]

    def png(self, tick, show_vision):
        row = self.row(tick)
        self.world.fill((42, 53, 43))
        self.vision.fill((0, 0, 0, 0))
        for obstacle in self.static['obstacles']:
            pygame.draw.rect(self.world, (128,128,128), pygame.Rect(*obstacle))
        roles = {3:(0,220,255),4:(255,235,40),5:(255,150,30),6:(230,220,120)}
        for kind in ('agents', 'predators'):
            for item in row[kind]:
                creature = object.__new__(Creature)
                ident, x, y, heading, energy = item[:5]
                traits = (self.catalog[str(ident)] if kind == 'agents' else
                          dict(size=10,color=(255,0,0),max_energy=200,
                               hearing_radius=60,vision_radius=250,cone_angle=math.pi/3))
                creature.__dict__.update(traits)
                creature.__dict__.update(x=x,y=y,direction=heading,energy=energy,
                                         _vision_poly=None if show_vision else [])
                creature.draw(self.world, self.vision, self.edges)
        if show_vision: self.world.blit(self.vision, (0,0))
        for ident,x,y,heading,energy,age,role in row['agents']:
            if role in roles: pygame.draw.circle(self.world, roles[role], (round(x),round(y)), 9, 2)
            self.world.blit(self.font.render(str(ident), True, (255,255,255)), (x+8,y+3))
        output = io.BytesIO()
        pygame.image.save(self.world, output, 'frame.png')
        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Benchmark root containing raw/pod-*/...')
    parser.add_argument('--port', type=int, default=9061)
    args = parser.parse_args()
    verify_sources()
    pygame.font.init()

    def cases():
        return sorted((json.loads(p.read_text()) for p in args.folder.glob('raw/pod-*/*/*/result.json')), key=lambda r:r['seed'])

    def folder(seed):
        paths = list(args.folder.glob(f'raw/pod-*/*/{seed:06d}/result.json'))
        if len(paths) != 1: raise FileNotFoundError(seed)
        return paths[0].parent

    @lru_cache(maxsize=1)
    def replay(seed): return Replay(folder(seed))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                mime = 'application/json'
                if url.path == '/':
                    payload = Path(__file__).with_suffix('.html').read_bytes()
                    mime = 'text/html; charset=utf-8'
                elif url.path == '/api/overview':
                    rows = cases()
                    payload = json.dumps(dict(cases=rows, summary=aggregate(rows))).encode()
                elif url.path in ('/api/case','/api/tick','/frame'):
                    seed = int(query['seed'][0])
                    path = folder(seed)
                    if url.path == '/api/case':
                        data = {key:json.loads((path/(key+'.json')).read_text()) for key in ('result','history','events','static')}
                        payload = json.dumps(data).encode()
                    else:
                        tick = int(query['tick'][0])
                        rep = replay(seed)
                        if url.path == '/frame':
                            payload = rep.png(tick, query.get('vision',['0'])[0]=='1')
                            mime = 'image/png'
                        else:
                            row = rep.row(tick)
                            payload = json.dumps(dict(row, catalog={str(a[0]):rep.catalog[str(a[0])] for a in row['agents']})).encode()
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control','no-store')
                self.end_headers(); self.wfile.write(payload)
            except (KeyError, ValueError, IndexError, FileNotFoundError): self.send_error(404)
            except BrokenPipeError: pass

    print(f'Entrapment benchmark: http://localhost:{args.port}', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
