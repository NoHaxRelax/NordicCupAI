"""Local review tool for sprite masks: verify, redraw, approve or reject, then export transparent sprites.

    python3 -m drone.grid_training.mask_editor serve            # http://localhost:8765
    python3 -m drone.grid_training.mask_editor shortlist        # pick 1 example per class x zoom x appearance
    python3 -m drone.grid_training.mask_editor export OUT_DIR   # transparent PNGs of approved/redrawn items

The shortlist holds one instance per (class, zoom, reference/validation appearance): the one whose
library mask fits best. The page shows only the shortlist unless you switch the set filter to all.

Items:
  lib:<variant>              library masks (one per class appearance)
  draw:<variant>             crops that still need a hand-drawn mask
  inst:<record_id>:<ann_idx> every training instance, mask transferred from the library

Decisions and redrawn masks are written to REVIEW_DIR (decisions.json + masks/). Nothing leaves this machine.
Only training-split tiles are read.
"""
import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

from . import sprite_library as sl

DATA = Path('data/drone/grid-comparison-20260918-v1/256')
LIBRARY = Path('data/drone/sprite-library-20260918-v4')
DRAW_KIT = Path('artifacts/drone-sprite-masks-to-draw-20260918')
REVIEW = Path('data/drone/sprite-mask-review-20260918')
HTML = Path(__file__).with_name('mask_editor.html')


def png_b64(img):
    ok, buf = cv2.imencode('.png', img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def safe(item_id):
    return item_id.replace(':', '__').replace('/', '_')


class Store:
    def __init__(self, data=DATA, library=LIBRARY, kit=DRAW_KIT, review=REVIEW):
        self.data, self.review = Path(data), Path(review)
        (self.review/'masks').mkdir(parents=True, exist_ok=True)
        self.m = json.loads((self.data/'manifest.json').read_text())
        self.rec = {r['id']: r for r in self.m['records']}
        self.lib_meta, self.lib = sl.load_library(library)
        self.lock = threading.Lock()
        self.auto_cache = {}
        dpath = self.review/'decisions.json'
        self.decisions = json.loads(dpath.read_text()) if dpath.exists() else {}
        spath = self.review/'shortlist.json'
        self.shortlist = {q['id']: q for q in json.loads(spath.read_text())['items']} if spath.exists() else {}
        self.items = {}
        for c, entries in self.lib.items():
            for e in entries:
                X, Y, U, V = e['crop_xyxy']
                self.items[f"lib:{e['variant']}"] = dict(id=f"lib:{e['variant']}", kind='library', class_name=c, variant=e['variant'],
                                                         tile=e['source_tile'], zoom=self.rec[e['source_tile']]['zoom'], crop=[X, Y, U, V])
        kit_file = Path(kit)/'masks_to_draw.json'
        if kit_file.exists():
            for q in json.loads(kit_file.read_text()):
                self.items[f"draw:{q['variant']}"] = dict(id=f"draw:{q['variant']}", kind='to_draw', class_name=q['class_name'], variant=q['variant'],
                                                          tile=q['tile'], zoom=self.rec[q['tile']]['zoom'], crop=q['crop_xyxy'],
                                                          kit_mask=str(Path(kit)/q['mask']))
        for r in self.m['records']:
            if r['split'] != 'train' or r['kind'] != 'positive':
                continue
            for i, a in enumerate(r['annotations']):
                if not a['fully_contained'] or a['class_name'] not in self.lib:
                    continue
                x, y, u, v = [int(round(t)) for t in a['bbox_xyxy']]
                if min(u-x, v-y) < 8:
                    continue
                pad = max(8, int(.35*max(u-x, v-y)))
                crop = [max(0, x-pad), max(0, y-pad), min(256, u+pad), min(256, v+pad)]
                iid = f"inst:{r['id']}:{i}"
                self.items[iid] = dict(id=iid, kind='instance', class_name=a['class_name'], variant=None, tile=r['id'], zoom=r['zoom'],
                                       crop=crop, box=[x, y, u, v], source=r['source'], frame=r['frame'])

    def listing(self):
        out = []
        for it in self.items.values():
            d = self.decisions.get(it['id'], {})
            sh = self.shortlist.get(it['id'])
            out.append(dict({k: it[k] for k in ('id', 'kind', 'class_name', 'variant', 'zoom', 'tile')}, source=it.get('source', self.rec[it['tile']]['source']),
                            status=d.get('status', 'unreviewed'), shortlist=sh is not None, score=sh['score'] if sh else None,
                            needs_drawing=bool(sh and sh['needs_drawing'])))
        order = {'to_draw': 0, 'library': 1, 'instance': 2}
        return sorted(out, key=lambda q: (order[q['kind']], q['class_name'], q['zoom'], q['id']))

    def tile(self, rid):
        return cv2.imread(str(self.data/self.rec[rid]['file']))

    def auto_mask(self, it, tile):
        """Library mask for library items; transferred mask for instances; kit or empty mask for to-draw items."""
        X, Y, U, V = it['crop']
        if it['kind'] == 'library':
            e = [e for e in self.lib[it['class_name']] if e['variant'] == it['variant']][0]
            return e['mask'].copy(), None, e['variant']
        if it['kind'] == 'to_draw':
            p = Path(it['kit_mask'])
            if p.exists():
                raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                return (cv2.resize(raw, (U-X, V-Y), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8), None, None
            return np.zeros((V-Y, U-X), np.uint8), None, None
        if it['id'] not in self.auto_cache:
            t = sl.transfer_best(tile, it['box'], it['zoom'], self.lib[it['class_name']])
            if t is None:
                self.auto_cache[it['id']] = dict(mask=np.zeros((V-Y, U-X), np.uint8), score=None, variant=None)
            else:
                self.auto_cache[it['id']] = dict(mask=t[0][Y:V, X:U].copy(), score=round(t[1], 3), variant=t[4])
        c = self.auto_cache[it['id']]
        return c['mask'].copy(), c['score'], c['variant']

    def current_mask(self, it, tile):
        auto, score, variant = self.auto_mask(it, tile)
        d = self.decisions.get(it['id'], {})
        if d.get('status') == 'redrawn' and (self.review/d['mask']).exists():
            return (cv2.imread(str(self.review/d['mask']), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8), auto, score, variant
        return auto, auto, score, variant

    def detail(self, iid):
        it = self.items[iid]
        tile = self.tile(it['tile'])
        X, Y, U, V = it['crop']
        mask, auto, score, variant = self.current_mask(it, tile)
        ctx = tile.copy()
        cv2.rectangle(ctx, (X, Y), (U-1, V-1), (0, 0, 255), 1)
        box = it.get('box')
        if box is None:
            anns = [a for a in self.rec[it['tile']]['annotations'] if a['class_name'] == it['class_name'] and a['fully_contained']]
            box = [int(round(t)) for t in anns[0]['bbox_xyxy']] if anns else None
        return dict(item=dict(it, **{'kit_mask': None}), status=self.decisions.get(iid, {}).get('status', 'unreviewed'),
                    note=self.decisions.get(iid, {}).get('note', ''), score=score, variant=variant,
                    crop=png_b64(tile[Y:V, X:U]), mask=png_b64(mask*255), auto=png_b64(auto*255), context=png_b64(ctx),
                    box_in_crop=None if box is None else [box[0]-X, box[1]-Y, box[2]-X, box[3]-Y], size=[U-X, V-Y])

    def save(self, iid, status, mask_b64=None, note=''):
        it = self.items[iid]
        assert status in ('approved', 'rejected', 'redrawn', 'unreviewed')
        entry = dict(status=status, kind=it['kind'], class_name=it['class_name'], tile=it['tile'], zoom=it['zoom'], crop_xyxy=it['crop'],
                     note=note, updated=time.strftime('%Y-%m-%dT%H:%M:%S'))
        if status == 'redrawn':
            raw = cv2.imdecode(np.frombuffer(base64.b64decode(mask_b64), np.uint8), cv2.IMREAD_UNCHANGED)
            # the page draws white-on-black with full opacity, so read a colour channel, never alpha
            m = raw if raw.ndim == 2 else raw[..., 0]
            X, Y, U, V = it['crop']
            assert m.shape == (V-Y, U-X), (m.shape, (V-Y, U-X))
            fill = float((m > 127).mean())
            if not 0 < fill < .97:
                raise ValueError(f'refusing a mask that covers {fill:.0%} of the crop')
            rel = f'masks/{safe(iid)}.png'
            cv2.imwrite(str(self.review/rel), ((m > 127)*255).astype(np.uint8))
            entry['mask'] = rel
        with self.lock:
            if status == 'unreviewed':
                self.decisions.pop(iid, None)
            else:
                self.decisions[iid] = entry
            tmp = self.review/'decisions.tmp'
            tmp.write_text(json.dumps(self.decisions, indent=1))
            tmp.replace(self.review/'decisions.json')
        return entry


def build_shortlist(store, min_score=.6, scores=None, exclude=None):
    """One instance per (class, zoom, source): highest transfer score, preferring boxes away from the tile edge.

    exclude: JSON list of {track_id, frames?: [..], reason} for labels found wrong on inspection; those
    instances are never picked (the dataset labels themselves are not changed here)."""
    skip = json.loads(Path(exclude).read_text()) if exclude else []
    def excluded(it):
        a = store.rec[it['tile']]['annotations'][int(it['id'].rsplit(':', 1)[1])]
        return any(a.get('track_id') == q['track_id'] and ('frames' not in q or it['frame'] in q['frames']) for q in skip)
    rows = []
    if scores:  # reuse a saved evaluation {class: [{tile, ann, score, zoom, src}]}
        for c, v in json.loads(Path(scores).read_text()).items():
            for t in v:
                rows.append((f"inst:{t['tile']}:{t['ann']}", t['score']))
    else:
        for iid, it in store.items.items():
            if it['kind'] == 'instance':
                t = sl.transfer_best(store.tile(it['tile']), it['box'], it['zoom'], store.lib[it['class_name']])
                rows.append((iid, -1 if t is None else round(t[1], 3)))
    best = {}
    for iid, score in rows:
        it = store.items.get(iid)
        if it is None or excluded(it):
            continue
        x, y, u, v = it['box']
        key = (it['class_name'], it['zoom'], it['source'])
        rank = (score, min(x, y, 256-u, 256-v) >= 6)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, iid)
    items = [dict(id=iid, class_name=k[0], zoom=k[1], source=k[2], score=r[0], needs_drawing=r[0] < min_score) for k, (r, iid) in sorted(best.items())]
    (store.review/'shortlist.json').write_text(json.dumps(dict(min_score=min_score, excluded=skip, items=items), indent=1))
    return items


def export(store, out):
    """Transparent BGRA sprite for every approved or redrawn item (rim-free edges, original pixels)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    index = []
    for iid, d in sorted(store.decisions.items()):
        if d['status'] not in ('approved', 'redrawn') or iid not in store.items:
            continue
        it = store.items[iid]
        tile = store.tile(it['tile'])
        mask, *_ = store.current_mask(it, tile)
        if mask.sum() < 4:
            continue
        X, Y, U, V = it['crop']
        hard = np.zeros(tile.shape[:2], np.uint8)
        hard[Y:V, X:U] = mask
        box = it.get('box') or [X, Y, U, V]
        rgba = sl.cutout_rgba(sl.sprite_patch(tile, box, hard))
        d_out = out/it['class_name']
        d_out.mkdir(exist_ok=True)
        f = d_out/f"{safe(iid)}.png"
        cv2.imwrite(str(f), rgba)
        index.append(dict(file=str(f.relative_to(out)), id=iid, class_name=it['class_name'], kind=it['kind'], zoom=it['zoom'], tile=it['tile'],
                          status=d['status'], size=list(rgba.shape[1::-1])))
    (out/'index.json').write_text(json.dumps(dict(review=str(store.review), sprites=index), indent=1))
    return index


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype='application/json'):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(b)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            try:
                if self.path in ('/', '/index.html'):
                    return self.send(200, HTML.read_bytes(), 'text/html; charset=utf-8')
                if self.path == '/api/items':
                    return self.send(200, store.listing())
                if self.path.startswith('/api/item/'):
                    iid = unquote(self.path[len('/api/item/'):])
                    if iid not in store.items:
                        return self.send(404, dict(error='unknown item'))
                    return self.send(200, store.detail(iid))
                self.send(404, dict(error='not found'))
            except Exception as e:  # keep the tool alive; show the error in the page
                self.send(500, dict(error=repr(e)))

        def do_POST(self):
            try:
                if not self.path.startswith('/api/item/'):
                    return self.send(404, dict(error='not found'))
                iid = unquote(self.path[len('/api/item/'):])
                if iid not in store.items:
                    return self.send(404, dict(error='unknown item'))
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send(200, store.save(iid, body['status'], body.get('mask'), body.get('note', '')))
            except Exception as e:
                self.send(500, dict(error=repr(e)))
    return H


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=8765)
    e = sub.add_parser('export')
    e.add_argument('out', type=Path)
    sh = sub.add_parser('shortlist')
    sh.add_argument('--scores', type=Path, default=None, help='saved transfer scores to reuse instead of recomputing (slow)')
    sh.add_argument('--exclude', type=Path, default=None, help='JSON list of {track_id, frames?, reason} to leave out')
    for q in (s, e, sh):
        q.add_argument('--library', type=Path, default=LIBRARY)
        q.add_argument('--review', type=Path, default=REVIEW)
    a = p.parse_args()
    store = Store(library=a.library, review=a.review)
    if a.cmd == 'shortlist':
        items = build_shortlist(store, scores=a.scores, exclude=a.exclude)
        print(f"{len(items)} shortlist items, {sum(q['needs_drawing'] for q in items)} need drawing -> {store.review/'shortlist.json'}")
        return
    if a.cmd == 'export':
        idx = export(store, a.out)
        print(f'exported {len(idx)} sprites to {a.out}')
        return
    print(f'{len(store.items)} items; review dir {store.review}; http://localhost:{a.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(store)).serve_forever()


if __name__ == '__main__':
    main()
