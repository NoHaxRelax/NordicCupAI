"""Render a served validation run as an H.264 video and a self-contained HTML player (one file, shareable).

What is drawn on the real flight (the 249 reconstructed 4K validation frames):
  green   the team's labels (the organisers' ground truth is not available to teams)
  cyan    objects our search found that the team never labelled (the three ta-ta were confirmed on the portal)
  red     the view the camera actually delivered to us at that frame, with its zoom level
  yellow  what we answered for the WHOLE frame at that step (class and confidence; thin below 0.25)
The HUD gives frame index, level, round-trip time, live tracks, and whether the answer was emitted or concealed.
Frames the portal never sent us (lost to latency) are marked.

    python elias/visualize_run.py --log elias/out/portal/F3_0_83/<uuid>.jsonl --tag F3_both_m1280
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GREEN, CYAN, RED, YELLOW, WHITE, GREY = (60, 220, 60), (230, 200, 40), (40, 40, 255), (40, 230, 255), (255, 255, 255), (150, 150, 150)


def found_objects():
    """frame -> [(class, box)] of the unlabelled objects, from the search chains named in validation_hidden2.json."""
    out = {}
    h2 = HERE/'data'/'validation_hidden2.json'
    if not h2.exists():
        return out
    tracks = json.loads(h2.read_text()).get('tracks', [])
    chains = {t: json.loads((HERE/'out'/f'unlabelled2_{t}.json').read_text()) for t in ('hel', 'both') if (HERE/'out'/f'unlabelled2_{t}.json').exists()}
    seen = set()
    for t in tracks:
        if t['what'] == 'dark blob' or t['search'] not in chains:
            continue
        for p in chains[t['search']][t['chain']]['pts']:
            key = (p[0], round(p[1]/30), round(p[2]/30))
            if key in seen:
                continue
            seen.add(key); out.setdefault(p[0], []).append((t['what'], [p[1]-p[3]/2, p[2]-p[4]/2, p[1]+p[3]/2, p[2]+p[4]/2]))
    return out


def label(img, text, x, y, colour, scale=0.45, thick=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(h+2, y)
    cv2.rectangle(img, (x, y-h-3), (x+w+2, y+2), (0, 0, 0), -1)
    cv2.putText(img, text, (x+1, y-1), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def render(frame_img, s, labels, found, row, index, n_frames, min_conf):
    img = cv2.resize(frame_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for a in labels:
        b = [int(v*s) for v in a['bbox']]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), GREEN, 1); label(img, a['object_id'], b[0], b[1]-2, GREEN, 0.38)
    for cls, bb in found:
        b = [int(v*s) for v in bb]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), CYAN, 1); label(img, cls+' (unlabelled)', b[0], b[3]+12, CYAN, 0.38)
    if row is None:
        label(img, f'frame {index}/{n_frames-1}   NO REQUEST REACHED US (frame lost to latency): scored as empty', 8, 22, RED, 0.6, 2)
    else:
        for r in sorted(row.get('response', []), key=lambda r: r['confidence']):
            if r['confidence'] < min_conf:
                continue
            b = r['bbox']; b = [int(b[0]*3840*s), int(b[1]*2160*s), int(b[2]*3840*s), int(b[3]*2160*s)]
            strong = r['confidence'] >= 0.25
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), YELLOW if strong else (20, 140, 160), 2 if strong else 1)
            if strong:
                label(img, f"{r['object_id']} {r['confidence']:.2f}", b[2]+2, b[1]+10, YELLOW, 0.38)
        rg = [int(v*s) for v in row['region']]
        cv2.rectangle(img, (rg[0], rg[1]), (rg[2], rg[3]), RED, 3)
        label(img, f"L{row['level']} view delivered to us", rg[0]+4, rg[1]+18, RED, 0.55, 2)
        emitted = row.get('emitted', True)
        hud = (f"frame {index}/{n_frames-1}   L{row['level']}   {row.get('total_ms', 0):.0f} ms   tracks {row.get('tracks', 0)}   "
               f"answered {sum(1 for r in row.get('response', []) if r['confidence'] >= 0.25)} boxes >= 0.25   "
               + ('EMITTED' if emitted else 'CONCEALED (answer withheld on purpose)'))
        label(img, hud, 8, 22, WHITE if emitted else GREY, 0.55, 1)
    y = img.shape[0]-8
    for text, col in (('team labels', GREEN), ('found by our search, unlabelled', CYAN), ('delivered view', RED), ('our answer for the whole frame', YELLOW)):
        label(img, text, 8, y, col, 0.42); y -= 18
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--width', type=int, default=1280); ap.add_argument('--html-width', type=int, default=960)
    ap.add_argument('--fps', type=float, default=3.0); ap.add_argument('--min-conf', type=float, default=0.05)
    ap.add_argument('--quality', type=int, default=58); ap.add_argument('--crf', type=int, default=27); ap.add_argument('--no-html', action='store_true')
    a = ap.parse_args()
    rows = {}
    for line in Path(a.log).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line); rows[int(r['frame_index'])] = r
    frames = sorted(p for p in (ROOT/'src'/'validation'/'images').glob('frame_*.png') if p.stat().st_size > 1_000_000)
    found = found_objects()
    out_dir = HERE/'out'/'viz'; out_dir.mkdir(parents=True, exist_ok=True)
    s = a.width/3840; hs = a.html_width/3840
    try:
        import imageio_ffmpeg
        ffm = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffm = None
    size = (int(3840*s), int(2160*s)); mp4 = out_dir/f'{a.tag}.mp4'
    if ffm:
        import subprocess
        proc = subprocess.Popen([ffm, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{size[0]}x{size[1]}', '-pix_fmt', 'bgr24',
                                 '-r', str(a.fps), '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(a.crf), '-preset', 'medium',
                                 '-movflags', '+faststart', str(mp4)], stdin=subprocess.PIPE)
        writer = None
    else:
        proc = None; writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*'mp4v'), a.fps, size)
    html_frames = []; lost = 0
    for p in frames:
        fr = int(p.stem.split('_')[1]); index = fr-1
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        labels = json.loads((ROOT/'src'/'validation'/'annotations'/f'{p.stem}.json').read_text())['annotations']
        row = rows.get(index); lost += row is None
        vid = render(img, s, labels, found.get(fr, []), row, index, len(frames), a.min_conf)
        if proc:
            proc.stdin.write(vid.tobytes())
        else:
            writer.write(vid)
        if not a.no_html:
            small = vid if hs == s else render(img, hs, labels, found.get(fr, []), row, index, len(frames), a.min_conf)
            ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            html_frames.append({'i': index, 'src': 'data:image/jpeg;base64,'+base64.b64encode(jpg.tobytes()).decode(),
                                'level': None if row is None else row['level'], 'ms': None if row is None else round(row.get('total_ms', 0)),
                                'emitted': None if row is None else bool(row.get('emitted', True)),
                                'guesses': [] if row is None else [[r['object_id'], round(r['confidence'], 2)] for r in sorted(row.get('response', []), key=lambda r: -r['confidence']) if r['confidence'] >= 0.25]})
    if proc:
        proc.stdin.close(); proc.wait()
    else:
        writer.release()
    print(f'{mp4}  ({mp4.stat().st_size/1e6:.1f} MB, {len(frames)} frames, {lost} never reached us)')
    if a.no_html:
        return
    page = out_dir/f'{a.tag}.html'
    title = html.escape(a.tag)
    page.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Drone flyby run {title}</title>
<style>body{{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}} .wrap{{max-width:{a.html_width}px;margin:0 auto;padding:8px}}
img{{width:100%;display:block;background:#000}} .bar{{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}}
input[type=range]{{flex:1;min-width:200px}} button,select{{background:#333;color:#eee;border:1px solid #555;padding:4px 10px;border-radius:4px;cursor:pointer}}
.legend span{{display:inline-block;margin-right:14px}} .sw{{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:-1px;border:2px solid}}
#g{{font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap;min-height:3em;color:#bbb}}</style></head><body><div class="wrap">
<h3 style="margin:6px 0">Drone flyby, validation flight, run {title}</h3>
<div class="legend"><span><i class="sw" style="border-color:#3cdc3c"></i>team labels</span><span><i class="sw" style="border-color:#28c8e6"></i>found by our search, unlabelled by the team</span>
<span><i class="sw" style="border-color:#ff2828"></i>view delivered to us (where we looked)</span><span><i class="sw" style="border-color:#ffe628"></i>our answer for the whole frame</span></div>
<img id="f"><div class="bar"><button id="pp">play</button><button id="prev">&lt;</button><button id="next">&gt;</button>
<input type="range" id="s" min="0" max="{len(html_frames)-1}" value="0"><span id="c"></span>
<select id="sp"><option value="3">3 fps (real time)</option><option value="1">1 fps</option><option value="6">6 fps</option><option value="12">12 fps</option></select></div>
<div id="g"></div>
<p style="color:#888">Arrow keys step, space plays. The organisers' ground truth is not available to teams; green boxes are the team's own labels. Yellow boxes are what the endpoint answered for the whole 3840x2160 frame at that step, including objects outside the red view, which it tracks by dead reckoning.</p>
</div><script>
const F={json.dumps(html_frames)};let i=0,t=null;const img=document.getElementById('f'),sl=document.getElementById('s'),c=document.getElementById('c'),g=document.getElementById('g'),pp=document.getElementById('pp'),sp=document.getElementById('sp');
function show(k){{i=(k+F.length)%F.length;const r=F[i];img.src=r.src;sl.value=i;c.textContent=(i+1)+' / '+F.length;
g.textContent=r.level===null?'no request reached us for this frame (lost to latency), scored as empty':('L'+r.level+'  '+r.ms+' ms  '+(r.emitted?'emitted':'concealed')+'\\n'+(r.guesses.length?r.guesses.map(x=>x[0]+' '+x[1].toFixed(2)).join('   '):'no boxes at 0.25 or above'));}}
function play(){{if(t){{clearInterval(t);t=null;pp.textContent='play';return}}pp.textContent='pause';t=setInterval(()=>show(i+1),1000/parseFloat(sp.value));}}
sp.onchange=()=>{{if(t){{clearInterval(t);t=setInterval(()=>show(i+1),1000/parseFloat(sp.value));}}}};pp.onclick=play;document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);
sl.oninput=()=>show(parseInt(sl.value));document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight')show(i+1);else if(e.key==='ArrowLeft')show(i-1);else if(e.key===' '){{e.preventDefault();play();}}}});show(0);
</script></body></html>''', encoding='utf-8')
    print(f'{page}  ({page.stat().st_size/1e6:.1f} MB, self-contained)')


if __name__ == '__main__':
    main()
