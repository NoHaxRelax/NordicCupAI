#!/usr/bin/env python3
"""Build a self-contained page for drawing the true box of each object the API probes cannot pin down.

Finding an extent by probing costs about 2.5 minutes an attempt and only ever answers one bit, so the classes
whose boxes sit near IoU 0.5 are cheaper to settle by eye. This crops the frames around each instance the plan
is weakest on, draws the current answer over the crop, and asks for the real one; it also opens a few search
panels where an object is predicted but nothing is detected.

Two frames per instance, early and late, so the box gives both the size and how it grows down the track.

    python3 make_box_tool.py --out artifacts/drone-box-tool-20260920/index.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / 'artifacts/drone-validation-box-review-20260919/backgrounds'
DISPLAY = 600

# classes whose AP says the boxes are close but not close enough, worst first
WEAK = ['large_launcher', 'large_tower', 'small_plane', 'tank']


def img_of(G, f, g):
    p = np.linalg.inv(G[f]) @ np.array([g[0], g[1], 1.0]); return p[:2] / p[2]


def crop_panel(f, cx, cy, half, box=None):
    """A square crop centred on (cx, cy), scaled to DISPLAY, plus the reference box in display coords."""
    path = FRAMES / f'frame_{f:06d}.jpg'
    if not path.exists():
        return None
    im = Image.open(path)
    half = min(half, W / 2, H / 2)
    cx = min(max(cx, half), W - half)          # slide the window inside the frame rather than padding it:
    cy = min(max(cy, half), H - half)          # a black band wastes the panel and hides the object
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    px0, py0 = max(0, -x0), max(0, -y0)
    sub = Image.new('RGB', (x1 - x0, y1 - y0), (18, 18, 20))
    sub.paste(im.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1))), (px0, py0))
    sub = sub.resize((DISPLAY, DISPLAY), Image.LANCZOS)
    buf = io.BytesIO(); sub.save(buf, 'JPEG', quality=88)
    k = DISPLAY / (x1 - x0)
    ref = None if box is None else [round((box[0] - x0) * k, 1), round((box[1] - y0) * k, 1),
                                    round((box[2] - x0) * k, 1), round((box[3] - y0) * k, 1)]
    return {'jpg': base64.b64encode(buf.getvalue()).decode(), 'ox': x0, 'oy': y0,
            'scale': round(k, 5), 'ref': ref}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--radius', type=float, default=90.0)
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())

    rows = defaultdict(list)
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['confidence'] >= 0.8 and int(f) in G:
                bb = [b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H]
                g = to_ground(G[int(f)], (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                rows[b['object_id']].append((int(f), g[0], g[1], bb))

    panels = []
    # Oscar spotted a second medium launcher at installation 2: the spiky crown that separates from the tower
    # body as the view goes oblique. These panels centre on the tower itself, so both the crown and the fork
    # are in shot, and ask the same question at the other two installations.
    # Both search questions are settled on the API and in the frames: the nine positions around large tower 3
    # scored exactly 0 over 249 delivered frames, and the frames 40-52 jet track is forest canopy, not an
    # aircraft. What is left is the extents, which is what the probes cannot settle cheaply.
    # 3. extents of the instances whose AP says the boxes are near the IoU 0.5 line
    for cls in WEAK:
        inst = []
        for f, gx, gy, bb in sorted(rows[cls]):
            for k in inst:
                if (k['x'] - gx) ** 2 + (k['y'] - gy) ** 2 <= a.radius ** 2:
                    n = k['n']; k['x'] = (k['x'] * n + gx) / (n + 1); k['y'] = (k['y'] * n + gy) / (n + 1)
                    k['n'] += 1; k['rows'].append((f, bb)); break
            else:
                inst.append({'x': gx, 'y': gy, 'n': 1, 'rows': [(f, bb)]})
        inst = [k for k in inst if k['n'] >= 8]
        for i, k in enumerate(sorted(inst, key=lambda d: min(f for f, _ in d['rows']))):
            k['rows'].sort()
            n = len(k['rows'])
            # the first and last frames of a track, not points a fifth of the way in: those two are where a
            # straight-line fit through the middle extrapolates worst, and where the remaining misses sit
            picks = [k['rows'][0], k['rows'][n // 2], k['rows'][-1]]
            for f, bb in picks:
                half = max(60.0, max(bb[2] - bb[0], bb[3] - bb[1]) * 3.0)
                p = crop_panel(f, (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, half, bb)
                if p:
                    p.update(kind='extent', cls=cls, inst=f'instance {i + 1} of {len(inst)} '
                             f'(frames {min(x for x, _ in k["rows"])}-{max(x for x, _ in k["rows"])})', frame=f,
                             note='Orange is what we answer now. Draw the box you think the organiser uses: '
                                  'tight around the object, including anything sticking out.')
                    panels.append(p)
    html = TEMPLATE.replace('__PANELS__', json.dumps(panels))
    Path(a.out).write_text(html)
    mb = Path(a.out).stat().st_size / 1e6
    print(f'{a.out}: {len(panels)} panels, {mb:.1f} MB')
    for kind in ('search', 'extent'):
        ks = [p for p in panels if p['kind'] == kind]
        print(f'  {kind}: {len(ks)}')
        for p in ks:
            print(f"    {p['cls']:16s} {p['inst']:44s} frame {p['frame']}")


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8"><title>Drone box review</title>
<style>
 :root { --bg:#111214; --fg:#e8e8ea; --dim:#9a9aa2; --acc:#4da3ff; --ref:#ff9d3c; }
 * { box-sizing:border-box }
 body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 ui-sans-serif,system-ui,sans-serif }
 header { padding:10px 16px; border-bottom:1px solid #2a2b30; display:flex; gap:16px; align-items:baseline; flex-wrap:wrap }
 h1 { font-size:16px; margin:0; font-weight:600 }
 #where { color:var(--acc); font-weight:600 }
 #note { color:var(--dim); padding:8px 16px; max-width:860px }
 main { display:flex; gap:20px; padding:0 16px 20px; align-items:flex-start; flex-wrap:wrap }
 #stage { position:relative; width:600px; height:600px; flex:0 0 auto; cursor:crosshair;
          background:#000; border:1px solid #2a2b30; border-radius:6px; overflow:hidden; touch-action:none }
 #stage img { position:absolute; inset:0; width:100%; height:100%; image-rendering:auto; user-select:none; -webkit-user-drag:none }
 .box { position:absolute; pointer-events:none }
 .ref { border:2px dashed var(--ref) }
 .draw { border:2px solid var(--acc); background:rgba(77,163,255,.12) }
 aside { flex:1 1 280px; min-width:260px }
 button { font:inherit; padding:8px 14px; margin:0 8px 8px 0; border-radius:6px; border:1px solid #3a3b42;
          background:#1c1d21; color:var(--fg); cursor:pointer }
 button:hover { border-color:var(--acc) }
 button.primary { background:var(--acc); color:#06131f; border-color:var(--acc); font-weight:600 }
 table { border-collapse:collapse; width:100%; font-size:13px; margin-top:12px }
 td { padding:2px 6px; border-top:1px solid #26272c; color:var(--dim) }
 td:first-child { color:var(--fg); white-space:nowrap }
 .done { color:#5ad18a } .absent { color:#ff7b7b }
 kbd { background:#26272c; border-radius:4px; padding:1px 6px; font-size:12px }
</style>
<header>
  <h1>Drone box review</h1>
  <span id="where"></span>
  <span id="count" style="color:var(--dim)"></span>
</header>
<p id="note"></p>
<div id="bar" style="padding:0 16px 10px">
  <button class="primary" id="save">Save box <kbd>enter</kbd></button>
  <button id="absent">No object here <kbd>x</kbd></button>
  <button id="prev">&larr; Prev <kbd>p</kbd></button>
  <button id="next">Next <kbd>n</kbd> &rarr;</button>
  <button id="clear">Clear</button>
</div>
<main>
  <div id="stage"><img id="shot" alt=""><div id="refbox" class="box ref" hidden></div><div id="drawbox" class="box draw" hidden></div></div>
  <aside>
    <p style="color:var(--dim);font-size:13px">Drag on the image to draw. Drag again to redo. The dashed
      orange box is the current answer, shown for reference only.</p>
    <button id="download">Download answers</button>
    <table id="log"></table>
  </aside>
</main>
<script>
const PANELS = __PANELS__;
const answers = {};
let i = 0, drag = null;
const stage = document.getElementById('stage'), shot = document.getElementById('shot');
const refbox = document.getElementById('refbox'), drawbox = document.getElementById('drawbox');

function show() {
  const p = PANELS[i];
  shot.src = 'data:image/jpeg;base64,' + p.jpg;
  document.getElementById('where').textContent = p.cls + ' — ' + p.inst + ' — frame ' + p.frame;
  document.getElementById('count').textContent = (i + 1) + ' / ' + PANELS.length;
  document.getElementById('note').textContent = p.note;
  if (p.ref) { place(refbox, p.ref); refbox.hidden = false; } else refbox.hidden = true;
  const a = answers[i];
  if (a && a.box) { place(drawbox, a.box); drawbox.hidden = false; } else drawbox.hidden = true;
  renderLog();
}
function place(el, b) {
  el.style.left = Math.min(b[0], b[2]) + 'px'; el.style.top = Math.min(b[1], b[3]) + 'px';
  el.style.width = Math.abs(b[2] - b[0]) + 'px'; el.style.height = Math.abs(b[3] - b[1]) + 'px';
}
function at(e) { const r = stage.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
stage.addEventListener('pointerdown', e => { drag = at(e); stage.setPointerCapture(e.pointerId);
  drawbox.hidden = false; place(drawbox, [drag[0], drag[1], drag[0], drag[1]]); });
stage.addEventListener('pointermove', e => { if (!drag) return; const q = at(e);
  place(drawbox, [drag[0], drag[1], q[0], q[1]]); });
stage.addEventListener('pointerup', e => { if (!drag) return; const q = at(e);
  const b = [Math.min(drag[0], q[0]), Math.min(drag[1], q[1]), Math.max(drag[0], q[0]), Math.max(drag[1], q[1])];
  drag = null;
  if (b[2] - b[0] < 4 || b[3] - b[1] < 4) { drawbox.hidden = true; return; }
  answers[i] = { box: b }; renderLog(); });

function full(p, b) {   // display px -> full-frame px
  return [p.ox + b[0] / p.scale, p.oy + b[1] / p.scale, p.ox + b[2] / p.scale, p.oy + b[3] / p.scale].map(v => Math.round(v * 10) / 10);
}
function renderLog() {
  const t = document.getElementById('log'); t.innerHTML = '';
  PANELS.forEach((p, k) => {
    const a = answers[k];
    const tr = t.insertRow();
    tr.insertCell().textContent = p.cls + ' f' + p.frame;
    const c = tr.insertCell();
    if (!a) c.textContent = '—';
    else if (a.absent) { c.textContent = 'no object'; c.className = 'absent'; }
    else { const F = full(p, a.box); c.textContent = Math.round(F[2] - F[0]) + '×' + Math.round(F[3] - F[1]) + ' px'; c.className = 'done'; }
    if (k === i) tr.style.background = '#1b1d22';
  });
}
document.getElementById('save').onclick = () => { if (answers[i] && answers[i].box) step(1); };
document.getElementById('absent').onclick = () => { answers[i] = { absent: true }; drawbox.hidden = true; step(1); };
document.getElementById('clear').onclick = () => { delete answers[i]; drawbox.hidden = true; renderLog(); };
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
function step(d) { i = Math.max(0, Math.min(PANELS.length - 1, i + d)); show(); }
document.getElementById('download').onclick = () => {
  const out = [];
  PANELS.forEach((p, k) => { const a = answers[k]; if (!a) return;
    out.push({ cls: p.cls, kind: p.kind, inst: p.inst, frame: p.frame,
               absent: !!a.absent, box: a.absent ? null : full(p, a.box) }); });
  const blob = new Blob([JSON.stringify({ answers: out }, null, 1)], { type: 'application/json' });
  const u = URL.createObjectURL(blob), el = document.createElement('a');
  el.href = u; el.download = 'box-answers.json'; el.click(); URL.revokeObjectURL(u);
};
addEventListener('keydown', e => {
  if (e.key === 'n' || e.key === 'ArrowRight') step(1);
  else if (e.key === 'p' || e.key === 'ArrowLeft') step(-1);
  else if (e.key === 'x') document.getElementById('absent').click();
  else if (e.key === 'Enter') document.getElementById('save').click();
});
show();
</script>
"""

if __name__ == '__main__':
    main()
