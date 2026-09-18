"""Evaluate completed classifiers on a fixed blind-proposal cohort."""
import argparse,datetime,os,subprocess,sys,time
from pathlib import Path
from common import read,sha
p=argparse.ArgumentParser()
for name in ['experiments','data','code','source_manifest','predictions']:
 p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
p.add_argument('--names',required=True);a=p.parse_args();run=Path(os.environ['NORDIC_RUN_DIR'])
for name in a.names.split(','):
 experiment=a.experiments/name;receipt=experiment/'result.json'
 while not receipt.exists():
  if (experiment/'failure.json').exists():raise SystemExit('Training failed: '+name)
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,9,18,8,30,tzinfo=datetime.timezone.utc):raise SystemExit('Evaluation window finished')
  time.sleep(20)
 r=read(receipt);assert r['state']=='completed';checkpoint=Path(r['checkpoint']);assert sha(checkpoint)==r['checkpoint_sha256']
 subprocess.run([sys.executable,str(a.code/'rerank_saved.py'),'--data',str(a.data),'--source-manifest',str(a.source_manifest),'--predictions',str(a.predictions),'--classifier',str(checkpoint),'--output',str(run/name)],check=True)
