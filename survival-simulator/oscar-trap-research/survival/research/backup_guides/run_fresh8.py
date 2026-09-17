"""Two-worker native fresh batch for the frozen burst policy."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/backup_guides/fresh8";STOP=Path("/tmp/predator-intake-stop")
EXPECTED="642725a431ee4778c1adcf100ffb45e7d85faf9c4e086f5923097f43202b7d9d"

def one(seed):
    if STOP.exists():return {"seed":seed,"stopped":True}
    logdir=OUT/"logs";logdir.mkdir(parents=True,exist_ok=True);log=logdir/f"map-{seed}.log"
    command=[sys.executable,str(ROOT/"research/simple_chase/run_streaming_v2.py"),
      "--policy","backup_guides.burst:BurstGuides","--map-seed",str(seed),
      "--fixture-seed",str(seed+10000),"--seconds","300","--predators","1",
      "--station-bait","--native-width","320"]
    env={**os.environ,"PYTHONPATH":f"{ROOT/'research'}:{ROOT/'debugger'}:{ROOT/'vendor/survival-simulator'}"}
    with log.open("w") as stream:
        done=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    return {"seed":seed,"returncode":done.returncode,"log":str(log.relative_to(ROOT))}

if __name__=="__main__":
    actual=hashlib.sha256((ROOT/"research/backup_guides/burst.py").read_bytes()).hexdigest()
    if actual!=EXPECTED:raise SystemExit(f"burst source changed: {actual}")
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(one,seed):seed for seed in range(10200,10208)}
        for future in as_completed(jobs):
            row=future.result();rows.append(row);print(json.dumps(row),flush=True)
    rows.sort(key=lambda r:r["seed"])
    (OUT/"launch-summary.json").write_text(json.dumps({"policy_sha256":EXPECTED,"workers":2,"rows":rows},indent=2)+"\n")
