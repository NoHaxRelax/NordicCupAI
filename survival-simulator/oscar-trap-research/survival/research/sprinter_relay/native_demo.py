"""Reproduce the reported prepared newborn takeover with native engine rendering."""
import hashlib,json
from pathlib import Path
from run import run,OUT
source=OUT/'replays.json'
old=json.loads(source.read_text())['runs'][1]
reproduction=dict(source_case='replays[1] native-rendered reproduction',source_file='replays.json',
    source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    limitations='New native-rendered reproduction of the saved configuration. The original state-only replay is preserved; this is not recovered original video.')
row=run(**old['parameters'],record='Prepared newborn takeover: native reproduction',
    record_every=10,native_render=True,reproduction=reproduction)
print(json.dumps({k:row[k] for k in ('run_id','replay','metrics_file','active_retention')},indent=2))
