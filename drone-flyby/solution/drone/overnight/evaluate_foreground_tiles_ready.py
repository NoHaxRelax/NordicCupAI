"""Compare binary tile localization with recognition against original multiclass truth."""
import datetime,os,subprocess,sys,time,json,hashlib
from pathlib import Path
w=Path(os.environ['NORDIC_WORK']);run=Path(os.environ['NORDIC_RUN_DIR']);folder=w/'runs/drone-foreground-tile-train/experiments/det-foreground-tiles-medium';receipt=folder/'result.json'
while not receipt.exists():
 if (folder/'failure.json').exists():raise SystemExit('Foreground tile training failed')
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,9,18,8,30,tzinfo=datetime.timezone.utc):raise SystemExit('Evaluation window finished')
 time.sleep(20)
r=json.loads(receipt.read_text());checkpoint=Path(r['checkpoint']);assert r['state']=='completed';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
for mode,extra in [('full',[]),('tiles',['--native-tiles'])]:
 subprocess.run([sys.executable,str(w/'code/analysis-v5/evaluate_views.py'),'--data','/tmp/nordic-drone-extra-views-v2/overnight-runpod-20260918-v2','--detector',str(checkpoint),'--classifier',str(w/'weights/proposal-conv/best.pt'),'--foreground','--output',str(run/mode),'--device','cpu',*extra],check=True)
