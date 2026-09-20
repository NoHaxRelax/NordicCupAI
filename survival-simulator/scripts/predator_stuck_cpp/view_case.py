"""Make a self-contained HTML viewer for one diagnostic event; no server needed."""
import argparse
import gzip
import json
from pathlib import Path

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predator confinement replay</title><style>
body{background:#101820;color:#e8eef2;font:16px system-ui;margin:0;padding:24px;max-width:1080px;margin:auto}
h1{font-size:24px;margin-bottom:8px}.muted{color:#a8b7c5}canvas{background:#172b30;width:100%;border-radius:10px}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(240px,1fr);gap:20px;margin:20px 0}
button{background:#254354;color:white;border:1px solid #567789;border-radius:6px;padding:9px 13px;cursor:pointer}
input[type=range]{width:100%;margin:20px 0}pre{white-space:pre-wrap;font:14px/1.65 ui-monospace,monospace}
@media(max-width:750px){.grid{grid-template-columns:1fr}body{padding:12px}}
</style><h1 id="title"></h1><div class="muted">Native 0.1-second steps · Amber: 15-unit test circle · Green: accepted move · Red: rejected move</div>
<div class="grid"><div><canvas id="world" width="840" height="700"></canvas>
<input id="slider" type="range" min="0" value="0"><div><button id="play">Play</button>
<button id="back">Previous tick</button> <button id="next">Next tick</button>
<button id="onset">Interval start</button> <button id="zoom">Show approach</button></div></div>
<div><pre id="state"></pre><p class="muted">Candidate lines show the actual attempted directions for this transition. A successful fallback can move in a different direction from the predator’s updated heading.</p>
<p id="follow" class="muted"></p></div></div>
<script>const data=__DATA__;
const rows=data.event.diagnostic_rows, cols=data.event.diagnostic_format.columns;
const ix=Object.fromEntries(cols.map((k,i)=>[k,i]));const e=data.event;
const canvas=document.getElementById('world'),ctx=canvas.getContext('2d'),slider=document.getElementById('slider');
slider.max=rows.length-1;let playing=false,wide=false,last=0;
document.getElementById('title').textContent=`Seed ${e.seed} · Predator ${e.predator_id} · ${e.biome}`;
document.getElementById('follow').textContent=e.follow_up.escaped_after_detection?
 `Left the original test circle at ${e.follow_up.first_exit_seconds}s.`:
 `No departure from the original test circle was observed before ${e.follow_up.observation_end_seconds}s.`;
function render(){const r=rows[+slider.value],v=k=>r[ix[k]],W=canvas.width,H=canvas.height;
 let cx=e.anchor[0],cy=e.anchor[1],span=110;
 if(wide){let xs=rows.map(r=>r[1]),ys=rows.map(r=>r[2]);let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);cx=(x0+x1)/2;cy=(y0+y1)/2;span=Math.max(110,x1-x0+60,(y1-y0+60)*W/H);}
 const scale=W/span,point=(x,y)=>[(x-cx)*scale+W/2,(y-cy)*scale+H/2];
 ctx.fillStyle='#172b30';ctx.fillRect(0,0,W,H);
 for(let i=0;i<data.map.obstacles.length;i++){const [x,y,w,h]=data.map.obstacles[i],p=point(x,y);ctx.fillStyle=i<4?'#48576a':'#57636d';ctx.fillRect(p[0],p[1],w*scale,h*scale);
 if(p[0]>-100&&p[0]<W&&p[1]>-100&&p[1]<H){ctx.fillStyle='#d0dae3';ctx.font='13px system-ui';ctx.fillText('obstacle '+i,p[0]+5,p[1]+18);}}
 const circle=(x,y,r,color,line=2)=>{const p=point(x,y);ctx.strokeStyle=color;ctx.lineWidth=line;ctx.beginPath();ctx.arc(...p,r*scale,0,Math.PI*2);ctx.stroke();};
 circle(...e.anchor,e.radius,'#ffc86e');
 ctx.strokeStyle='#78aab7';ctx.lineWidth=1.5;ctx.beginPath();for(let i=0;i<=+slider.value;i++){const p=point(rows[i][1],rows[i][2]);i?ctx.lineTo(...p):ctx.moveTo(...p);}ctx.stroke();
 const from=point(v('before_x'),v('before_y')),tests=v('candidate_tests'),accepted=v('accepted_candidate');
 for(let n=0;n<tests;n++){let angle=v('absolute_move_direction');if(n){const i=n-1;angle+=(Math.PI/18)*Math.floor((i+1)/2)*(i%2?-1:1);}
 const p=point(v('before_x')+v('scaled_distance')*Math.cos(angle),v('before_y')+v('scaled_distance')*Math.sin(angle));
 ctx.strokeStyle=n===accepted?'#54e6a9':'#ef6e7899';ctx.lineWidth=n===accepted?3:1;ctx.beginPath();ctx.moveTo(...from);ctx.lineTo(...p);ctx.stroke();}
 circle(r[1],r[2],e.predator_size,'#f1e7d2',2);const p=point(r[1],r[2]),q=point(r[1]+12*Math.cos(r[3]),r[2]+12*Math.sin(r[3]));
 ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();
 const mode=e.diagnostic_format.modes[String(v('mode'))]||'birth';
 document.getElementById('state').textContent=`Time: ${(r[0]*.1).toFixed(1)} s\nInterval: ${e.start_seconds}–${e.detected_seconds} s\n\nDecision: ${mode}\nPosition: ${r[1].toFixed(3)}, ${r[2].toFixed(3)}\nEnergy: ${r[4].toFixed(3)}\nResting: ${!!r[5]}\nTurn: ${(v('turn')*180/Math.PI).toFixed(3)}°\nStep distance: ${v('scaled_distance').toFixed(3)}\nEdge clearance: ${v('edge_clearance').toFixed(3)}\nVisible edges: ${v('visible_edges')}\nCandidate tests: ${tests}\nAccepted candidate: ${accepted}\nFirst blocker: ${v('first_blocking_obstacle')}`;
}
function onset(){slider.value=rows.findIndex(r=>r[0]===e.samples[0][0]);render();}
slider.oninput=render;document.getElementById('onset').onclick=onset;
document.getElementById('back').onclick=()=>{slider.value=Math.max(0,+slider.value-1);render()};
document.getElementById('next').onclick=()=>{slider.value=Math.min(rows.length-1,+slider.value+1);render()};
document.getElementById('zoom').onclick=()=>{wide=!wide;document.getElementById('zoom').textContent=wide?'Focus on trap':'Show approach';render()};
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play'};
function animate(t){if(playing&&t-last>=100){slider.value=(+slider.value+1)%rows.length;render();last=t;}requestAnimationFrame(animate)}
onset();requestAnimationFrame(animate);
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with gzip.open(args.trace,'rt',encoding='utf-8') as f:
        event=json.load(f)
    geometry=json.loads((args.trace.parent.parent/'map.json').read_text())
    payload=json.dumps(dict(event=event,map=geometry),separators=(',',':')).replace('<','\\u003c')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(PAGE.replace('__DATA__',payload),encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()
