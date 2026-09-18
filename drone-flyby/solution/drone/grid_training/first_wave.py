"""Two sequential runs per GPU; stop on any failure and retain all outputs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--size',type=int,choices=[256,384],required=True);a=p.parse_args()
 root=Path(os.environ['NORDIC_RUN_DIR']);plan=[]
 for pretrained in [False,True]:
  name=f'resnet18-{a.size}-'+('pretrained' if pretrained else 'scratch')
  cmd=[sys.executable,'-m','drone.grid_training.train','--data',str(Path(a.data)/str(a.size)),'--output',str(root/name),'--arch','resnet18','--epochs','24','--steps','50','--batch','32']
  if pretrained:cmd.append('--pretrained')
  plan.append(dict(name=name,command=cmd))
 (root/'plan.json').write_text(json.dumps(plan,indent=2))
 for row in plan:
  print('START',row['name'],flush=True);subprocess.run(row['command'],check=True)
 (root/'wave-complete.json').write_text(json.dumps(dict(completed=time.time(),experiments=[r['name'] for r in plan])))
if __name__=='__main__':main()
