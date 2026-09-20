"""Local durable progress; safe to run while the coordinator is active."""
import argparse,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');ap.add_argument('--interval',type=int,default=30);a=ap.parse_args();out=ROOT/'docs/selfstuck5/run'
 try:
  while True:
   print('\n'+time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
   for name in ['progress.json','complete.json','ERROR.json']:
    p=out/name
    if p.exists():print(name+': '+p.read_text().strip(),flush=True)
   for p in sorted(out.glob('*-trials.json')):
    r=json.loads(p.read_text());print(f'{p.stem}: {len(r)}/50, best {max(x["score"]for x in r):.1f}',flush=True)
   if a.once or(out/'complete.json').exists()or(out/'ERROR.json').exists():break
   time.sleep(a.interval)
 except KeyboardInterrupt:pass
if __name__=='__main__':main()
