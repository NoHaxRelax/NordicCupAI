from pathlib import Path
import gzip,json
ROOT=Path(__file__).resolve().parents[2]
rows=[]
for p in (ROOT/'results/depth5_test').rglob('*.json'):
    r=json.loads(p.read_text())
    if not isinstance(r,dict) or 'replay' not in r:continue
    data=json.loads(gzip.decompress((ROOT/r['replay']).read_bytes()))
    frames=data['frames'];expected=round(r['seconds']*10)+1
    assert len(frames)==expected,(p,len(frames),expected)
    assert all('native_image' in f for f in frames)
    assert all(abs(f['t']-i/10)<.001 for i,f in enumerate(frames))
    chunk_dir=Path(__file__).parent/'frames'/p.stem
    chunk_dir.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(frames),100):
        chunk=chunk_dir/f'{start//100}.json.gz'
        if not chunk.exists():chunk.write_bytes(gzip.compress(json.dumps([{'t':f['t'],'native_image':f['native_image']} for f in frames[start:start+100]]).encode()))
    depth=r.get('depth',r.get('bait_depth',5))
    bait=next((a for a in frames[-1]['agents'] if a['id']==0),None)
    mode='guided' if 'follower_gap' in r else 'direct'
    rows.append(dict(mode=mode,reason=data['summary'].get('reason'),chunks='frames/'+p.stem+'/',title=f"Depth {depth:g} · gap {r['gap']:g} · {mode} {r.get('arrivals',r['predators'])}/{r['predators']} · {r['seconds']:g}s",depth=depth,gap=r['gap'],seed=r['seed'],duration=r['seconds'],requested=r.get('requested_seconds'),frames=len(frames),bait_alive=bool(bait),acquired=r['acquired'],final=r['final_joint'],losses=r['physical_losses'],predators=r['predators'],replay='/'+r['replay'],receipt='/'+str(p.relative_to(ROOT)),deaths=r['deaths']))
rows.sort(key=lambda r:(r['mode']!='guided',-r['duration'],r['gap'],r['depth']))
(Path(__file__).parent/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'native_frames':sum(r['frames'] for r in rows),'all_frames_verified':True}))
