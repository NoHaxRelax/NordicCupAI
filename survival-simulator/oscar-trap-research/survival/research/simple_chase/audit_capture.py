"""Post-hoc audit from immutable native replay/target traces; never controller input."""
from pathlib import Path
import json,math,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from debugger.replay_stream import iter_frames

def audit(path):
 r=json.loads(path.read_text());switches=[s for s in r['target_switches'] if s['tracked_predator_slot']==0]
 index=0;target=None;active_seen=False;streak=0;start=None;confirmed=None;losses=[];last_alive=None;death_snapshot=None;tracked=None;first_bait=None
 death=next((d['time'] for d in r['deaths'] if d['agent_id']==r['guide_id']),None)
 for f in iter_frames(ROOT/r['replay']):
  t=f['t']
  if tracked is None and f['predators']:tracked=f['predators'][0]['id']
  p=next((p for p in f['predators'] if p['id']==tracked),None)
  a=next((a for a in f['agents'] if a['id']==r['guide_id']),None)
  if a:last_alive={'time':t,'guide_distance_to_mouth':math.dist((a['x'],a['y']),r['site']['mouth']),'predator_distance_to_mouth':None if p is None else math.dist((p['x'],p['y']),r['site']['mouth'])}
  elif death_snapshot is None:death_snapshot=last_alive
  while index<len(switches) and switches[index]['time']<=t+.001:
   target=switches[index]['target'];index+=1
   if target==r['bait_id'] and first_bait is None:first_bait=switches[index-1]['time']
  if p is None:physical=False
  else:
   delta=[p[k]-r['site']['mouth'][i] for i,k in enumerate(('x','y'))]
   physical=math.hypot(*delta)<=75.001 and sum(x*y for x,y in zip(delta,r['site']['inward']))<=10.001
  if physical and target==r['bait_id']:active_seen=True
  eligible=physical and (target==r['bait_id'] or target=='resting' and active_seen)
  if eligible:
   if streak==0:start=t
   streak+=1
   if streak>=20 and confirmed is None:confirmed=start
  else:
   streak=0;active_seen=False
   if confirmed is not None and not losses:losses.append({'time':t,'physical':physical,'target':target})
 distant=(death is not None and first_bait is not None and first_bait-death>3 and death_snapshot is not None and (death_snapshot['predator_distance_to_mouth'] or 0)>200)
 result={'schema':'posthoc-active-bait-audit-v1','receipt':str(path.relative_to(ROOT)),'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'original_harness_success':r['success'],'first_active_bait_target':first_bait,'confirmed_capture_start':confirmed,'first_loss_after_confirmed_capture':losses,'recorded_seconds':r['seconds'],'seconds_after_capture':None if confirmed is None else round(r['seconds']-confirmed,1),'guide_death':death,'last_alive_snapshot':death_snapshot,'distant_autonomous_capture':distant,'active_bait_30s_hold':confirmed is not None and not losses and r['seconds']-confirmed>=30 and r['bait_alive'],'caveat':'Post-hoc rounded replay audit. Removes rest-only provisional acquisition; flags clear distant post-death captures, but does not prove causal attribution for every survivor. Original receipts remain unchanged.'}
 return result

if __name__=='__main__':
 out=ROOT/'results/simple_chase/active_bait_audits';out.mkdir(exist_ok=True)
 for name in sys.argv[1:]:
  path=Path(name);path=path if path.is_absolute() else ROOT/path
  result=audit(path);dest=out/(path.stem+'.audit.json')
  dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('receipt','original_harness_success','active_bait_30s_hold','distant_autonomous_capture','confirmed_capture_start')}),flush=True)
