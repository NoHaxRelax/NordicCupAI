"""Wait for an atomically completed experiment, verify its checkpoint, then evaluate."""
import argparse,datetime,json,os,subprocess,sys,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--classifier',type=Path);p.add_argument('--foreground',action='store_true');a=p.parse_args()
run=Path(os.environ['NORDIC_RUN_DIR']);receipt=a.experiment/'result.json'
while not receipt.exists():
 if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,9,18,8,30,tzinfo=datetime.timezone.utc):raise SystemExit('Evaluation window finished')
 time.sleep(20)
r=json.loads(receipt.read_text());assert r['state']=='completed';checkpoint=Path(r['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
cmd=[sys.executable,str(a.code/'evaluate_views.py'),'--data',str(a.data),'--detector',str(checkpoint),'--output',str(run/'evaluation'),'--device','cpu']
if a.classifier:cmd+=['--classifier',str(a.classifier)]
if a.foreground:cmd+=['--foreground']
subprocess.run(cmd,check=True)
