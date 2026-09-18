"""Run one bounded follow-up only after the named predecessor exits successfully."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--data',required=True);a=p.parse_args();deadline=time.monotonic()+1800
 while not a.predecessor.exists():
  if time.monotonic()>deadline:raise RuntimeError('Predecessor timed out; no follow-up started')
  time.sleep(10)
 if json.loads(a.predecessor.read_text())['exit_code']!=0:raise RuntimeError('Predecessor failed; no follow-up started')
 root=Path(os.environ['NORDIC_RUN_DIR'])
 cmd=[sys.executable,'-m','drone.grid_training.train','--data',a.data,'--output',str(root/'model'),'--arch','resnet18','--epochs','60','--patience','60','--steps','50','--batch','32','--lr','0.0003']
 subprocess.run(cmd,check=True)
 subprocess.run([sys.executable,'-m','drone.grid_training.diagnose','--data',a.data,'--run',str(root/'model')],check=True)
if __name__=='__main__':main()
