"""Summarize immutable receipts without loading replay payloads."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/simple_chase'
rows=[]
for path in sorted(OUT.glob('*.json')):
 if path.name.endswith('.correction.json'):continue
 row=json.loads(path.read_text())
 if not row.get('policy') or not row.get('schema','').startswith('real-map-intake-result'):continue
 rows.append({'receipt':path.name,'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              **{k:row.get(k) for k in ('policy','policy_hash','harness_hashes','map_seed','fixture_seed','seconds','requested_seconds','reason','success','guided_delivery_times','deaths','bait_alive','guide_alive','physical_losses_after_delivery','target_losses_after_delivery','arranged_aligned_final_approach','outcome','replay','replay_frames')}})
(OUT/'receipt-summary.json').write_text(json.dumps({'setup':'Predeployed depth5 bait, guide55 units from initially awake predator; subsequent actions use native observations only. Aligned controls explicitly separate.','rows':rows},indent=2)+'\n')
print('Summarized',len(rows),'receipts')
