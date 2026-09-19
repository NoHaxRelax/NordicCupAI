"""Browser replay of every recorded native-game tick, rendered by native draw()."""
import argparse
from functools import lru_cache
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse, parse_qs

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pygame
from src.elements.environment import Environment
from src.elements.creature import Creature
from src.elements.fruit import Fruit
from src.elements.tree import Tree


def restore(cls, data):
    obj = object.__new__(cls)
    obj.__dict__.update(data)
    return obj


class Replay:
    def __init__(self, folder):
        self.folder = folder
        static = json.loads((folder/'static.json').read_text())
        self.env = object.__new__(Environment)
        self.env.width, self.env.height, self.env.edges = static['width'], static['height'], static['edges']
        self.env.static_surface = pygame.image.load(folder/'background.png')
        for key in ('shadow_surface', 'obstacle_surface', 'world_surface', 'vision_screen', 'leaf_screen'):
            setattr(self.env, key, pygame.Surface((self.env.width, self.env.height), pygame.SRCALPHA))
        self.screen = pygame.Surface((1200, round(1200*self.env.height/self.env.width)))

    @lru_cache(maxsize=3)
    def chunk(self, index):
        with gzip.open(self.folder/'chunks'/f'{index:05d}.json.gz', 'rt') as handle: return json.load(handle)

    def row(self, tick): return self.chunk(tick//100)[tick%100]

    @lru_cache(maxsize=64)
    def png(self, tick):
        row = self.row(tick)
        for key, cls in (('agents', Creature), ('predators', Creature), ('fruits', Fruit), ('trees', Tree)):
            setattr(self.env, key, [restore(cls, data) for data in row['world'][key]])
        self.env.draw(self.screen)
        # Role rings identify agents while preserving the native game rendering.
        colors = dict(guide=(0, 220, 255), bait=(255, 235, 40), replacement_bait=(255, 150, 30),
                      retired_bait=(230, 220, 120), bait_candidate=(160, 100, 255))
        scale = self.screen.get_width()/self.env.width
        font = pygame.font.SysFont('monospace', 12)
        for a in self.env.agents:
            role = row['policy']['roles'].get(str(a.agent_id), '')
            x, y = round(a.x*scale), round(a.y*scale)
            if role in colors: pygame.draw.circle(self.screen, colors[role], (x, y), 9, 2)
            self.screen.blit(font.render(str(a.agent_id), True, (255, 255, 255)), (x+7, y+3))
        output = io.BytesIO()
        pygame.image.save(self.screen, output, 'replay.png')
        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--port', type=int, default=9059)
    args = parser.parse_args()
    pygame.font.init()
    replay = Replay(args.folder)
    render_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/':
                    payload = (Path(__file__).with_suffix('.html')).read_bytes()
                    mime = 'text/html; charset=utf-8'
                elif url.path == '/summary':
                    payload = (args.folder/'summary.json').read_bytes()
                    mime = 'application/json'
                elif url.path in ('/frame', '/tick'):
                    tick = int(parse_qs(url.query)['tick'][0])
                    if url.path == '/frame':
                        with render_lock:
                            payload, mime = replay.png(tick), 'image/png'
                    else:
                        row = replay.row(tick)
                        payload = json.dumps({k: v for k, v in row.items() if k != 'world'}).encode()
                        mime = 'application/json'
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers(); self.wfile.write(payload)
            except (ValueError, KeyError, IndexError, FileNotFoundError): self.send_error(404)
            except BrokenPipeError: pass

    print(f'Native game replay: http://localhost:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
