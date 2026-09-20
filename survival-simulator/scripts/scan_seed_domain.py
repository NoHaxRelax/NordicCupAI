"""Parallel, restartable seed search; covers every seed in [start,end) once.

Only public terrain samples are inputs. Worker output includes domain/elapsed
metadata so partial searches cannot masquerade as complete recovery.
"""
import argparse,concurrent.futures,json,pathlib,subprocess,time,os,fcntl,hashlib
p=argparse.ArgumentParser()
p.add_argument('--scanner',type=pathlib.Path,required=True)
p.add_argument('--samples',type=pathlib.Path,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
p.add_argument('--workers',type=int,default=32)
p.add_argument('--start',type=int,default=0)
p.add_argument('--end',type=int,default=2**32)
p.add_argument('--chunk',type=int,default=2**22)
p.add_argument('--deadline-seconds',type=float,default=540,
               help='Search time budget; caller must reserve time within the 600-second total for collection/replay.')
args=p.parse_args()
args.scanner=args.scanner.resolve();args.samples=args.samples.resolve()
args.out.mkdir(parents=True,exist_ok=True)
lock=(args.out/'coordinator.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=dict(start=args.start,end=args.end,chunk=args.chunk,
    scanner_sha256=hashlib.sha256(args.scanner.read_bytes()).hexdigest(),
    samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest())
manifest_path=args.out/'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
    raise SystemExit('Search inputs differ: use a new output directory')
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
(args.out/'coordinator.pid').write_text(str(os.getpid())+'\n')
start=time.monotonic();deadline=start+args.deadline_seconds
def job(bounds):
    lo,hi=bounds;path=args.out/f'{lo}-{hi}.json'
    if path.exists():return json.loads(path.read_text())
    remaining=deadline-time.monotonic()
    if remaining<=0:return None
    try:
        proc=subprocess.run([str(args.scanner),str(args.samples),str(lo),str(hi)],capture_output=True,text=True,check=True,timeout=remaining)
    except subprocess.TimeoutExpired:
        return None
    result=json.loads(proc.stderr)
    result['seeds']=list(map(int,proc.stdout.split()))
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(path)
    return result
bounds=[(lo,min(args.end,lo+args.chunk)) for lo in range(args.start,args.end,args.chunk)]
done=0;candidates=[];cpu=0
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    for r in pool.map(job,bounds):
        if r is None:continue
        done+=r['end']-r['start'];candidates.extend(r['seeds']);cpu+=r['seconds']
        status=dict(start=args.start,end=args.end,checked=done,total=args.end-args.start,
                    candidates=len(candidates),seconds=time.monotonic()-start,worker_seconds=cpu,
                    complete=done==args.end-args.start)
        tmp=args.out/'progress.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(args.out/'progress.json')
        print(json.dumps(status),flush=True)
if done==args.end-args.start:
    (args.out/'candidates.json').write_text(json.dumps(sorted(candidates)))
else:
    status=dict(start=args.start,end=args.end,checked=done,total=args.end-args.start,
                candidates=len(candidates),seconds=time.monotonic()-start,worker_seconds=cpu,
                complete=False,deadline_exceeded=True)
    (args.out/'progress.json').write_text(json.dumps(status,indent=2))
    raise SystemExit('Seed search deadline exceeded; use observation-only fallback')
