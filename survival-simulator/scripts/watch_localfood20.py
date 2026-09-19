"""Read-only progress dashboard for the late-game experiment; Ctrl-C to exit."""
import argparse
import concurrent.futures
import json
import pathlib
import shlex
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = r'''
import json,pathlib
r=pathlib.Path('/workspace/lucas-localfood20')
i=INDEX
t=r/'results'/('pod-'+str(i)); e=r/'final'/('shard-'+str(i))
trials=[]
for p in sorted(t.glob('*-trials.json')):
    try:
        rows=json.loads(p.read_text())
        trials.append((p.name.removesuffix('-trials.json'),len(rows),max(x['score'] for x in rows)))
    except (ValueError,KeyError):pass
n=sum(x[1] for x in trials)
phase='preparing/verifying checkpoints'
if (t/'checkpoints.json').exists():phase='200 checkpoints verified'
if trials:phase='; '.join(f'{name} {count}/32 best gain={best:.1f}' for name,count,best in trials)
if (t/'complete.json').exists():phase='BO complete; awaiting final evaluation'
games=0
if (e/'games.jsonl').exists():
    with (e/'games.jsonl').open() as f:games=sum(1 for line in f if line.endswith('\n'))
    phase=f'final evaluation {games}/2300 games'
if (e/'complete.json').exists():phase='final evaluation complete'
errors=[]
for log in ('run.log','final.log'):
    p=r/log
    if p.exists():
        tail=p.read_text()[-6000:]
        if 'Traceback (most recent call last)' in tail:errors.append(log+': '+tail.strip().splitlines()[-1])
if errors:phase='ERROR '+ '; '.join(errors)
print(json.dumps(dict(trials=n,games=games,phase=phase)))
'''

def check(p, key):
    command='python3 -c '+shlex.quote(REMOTE.replace('INDEX',str(p['index'])))
    try:
        result=subprocess.run(['ssh','-i',key,'-o','BatchMode=yes','-o','UpdateHostKeys=no',
            '-o','ConnectTimeout=5','-p',str(p['port']),f"root@{p['host']}",command],
            capture_output=True,text=True,timeout=15)
        if result.returncode:
            return dict(phase='SSH error: '+result.stderr.strip()[-200:])
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired,ValueError) as exc:
        return dict(phase=str(exc))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--once',action='store_true')
    ap.add_argument('--interval',type=float,default=60)
    ap.add_argument('--key',default=str(pathlib.Path.home()/'.ssh/runpod_codex_team'))
    args=ap.parse_args()
    pods=json.loads((ROOT/'docs/localfood20/dispatch.json').read_text())
    try:
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                statuses=list(pool.map(lambda p:check(p,args.key),pods))
            print('\n'+time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
            for p,s in zip(pods,statuses):print(f"Pod {p['index']:2} | {s['phase']}",flush=True)
            print(f"BO: {sum(s.get('trials',0) for s in statuses)}/640 trials | "
                  f"Final: {sum(s.get('games',0) for s in statuses)}/23000 games",flush=True)
            if args.once:break
            time.sleep(max(1,args.interval))
    except KeyboardInterrupt:pass

if __name__=='__main__':main()
