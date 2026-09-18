"""Evaluate completed tile-trained detectors using full and blind tiled views."""
import datetime,os,subprocess,sys,time,json,hashlib
from pathlib import Path
w=Path(os.environ['NORDIC_WORK']);run=Path(os.environ['NORDIC_RUN_DIR'])
for name in ['det-tiles-medium','det-tiles-large']:
 folder=w/'runs/drone-night-detector-tiles/experiments'/name;receipt=folder/'result.json'
 while not receipt.exists():
  if (folder/'failure.json').exists():raise SystemExit('Training failed: '+name)
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,9,18,8,30,tzinfo=datetime.timezone.utc):raise SystemExit('Evaluation window finished')
  time.sleep(20)
 result=json.loads(receipt.read_text());checkpoint=Path(result['checkpoint']);assert result['state']=='completed';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==result['checkpoint_sha256']
 for mode,extra in [('full',[]),('tiles',['--native-tiles'])]:
  subprocess.run([sys.executable,str(w/'code/analysis-v5/evaluate_views.py'),'--data','/tmp/nordic-drone-extra-views-v2/overnight-runpod-20260918-v2','--detector',str(checkpoint),'--output',str(run/(name+'-'+mode)),'--device','cpu',*extra],check=True)
