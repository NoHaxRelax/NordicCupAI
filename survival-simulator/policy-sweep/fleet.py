"""Provision a small RunPod fleet, bootstrap each box, and launch one arm-set per pod.

Every arm runs the SAME 1000 seeds, so all comparisons are paired. Pods are independent;
a pod that fails to come up just loses its arms rather than the run.

  python fleet.py provision 7          # bring up N pods and bootstrap them
  python fleet.py launch               # assign arm-sets and start sweeps
  python fleet.py status               # rows completed per pod
  python fleet.py collect              # pull all csvs locally
  python fleet.py kill                 # TERMINATE EVERYTHING
"""
import json
import pathlib
import subprocess
import sys
import time
import tomllib

import runpod

runpod.api_key = tomllib.loads((pathlib.Path.home() / ".runpod" / "config.toml").read_text())["apikey"]
STATE = pathlib.Path("C:/Users/nikol/AppData/Local/Temp/fleet.json")
PUB = (pathlib.Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
SAP = "C:/Users/nikol/AppData/Local/Temp/sap.tar"
BOOT = "C:/Users/nikol/AppData/Local/Temp/bootstrap_pod.sh"
SSHOPT = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
          "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=30"]

# The GPU is irrelevant -- this is a CPU-bound benchmark. Take whatever has capacity.
CANDS = [("NVIDIA GeForce RTX 4090", "SECURE"), ("NVIDIA RTX A4000", "SECURE"),
         ("NVIDIA RTX A5000", "SECURE"), ("NVIDIA GeForce RTX 3090", "SECURE"),
         ("NVIDIA A40", "SECURE"), ("NVIDIA L4", "SECURE"),
         ("NVIDIA RTX A4000", "COMMUNITY"), ("NVIDIA GeForce RTX 4090", "COMMUNITY")]

# Arm sets, one per pod. Pod 1 (already running) has 0,46,44,45,47,48,52,53,50,35,39,49.
ARM_SETS = [
    [27, 36, 37],        # predator-related rejections (16-seed rejects)
    [40, 41, 42],        # exact-age lifecycle rejections
    [51, 197, 198],      # early fruit + future-fruit forecasting
    [5, 6, 12],          # activation timing + breed reserve
    [14, 24, 25],        # breed reserve + tree/fruit reach multipliers
    [7, 8, 9],           # model-state modes
    [100, 140, 160],     # higher birth-forecast variants
]
SEEDS = 1000


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"pods": []}


def save(s):
    STATE.write_text(json.dumps(s, indent=1))


def ssh(p, cmd, timeout=900):
    return subprocess.run(["ssh", *SSHOPT, "-p", str(p["port"]), f"root@{p['ip']}", cmd],
                          capture_output=True, text=True, timeout=timeout)


def provision(n):
    s = load()
    for k in range(n):
        pod = None
        for gpu, cloud in CANDS:
            try:
                pod = runpod.create_pod(
                    name=f"policy-sweep-{k+2}",
                    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                    gpu_type_id=gpu, gpu_count=1, cloud_type=cloud,
                    container_disk_in_gb=40, volume_in_gb=0, ports="22/tcp",
                    env={"PUBLIC_KEY": PUB}, start_ssh=True)
                print(f"pod {k+2}: created on {gpu} ({cloud}) {pod['id']}", flush=True)
                break
            except Exception as ex:
                print(f"  {gpu} {cloud}: {str(ex)[:60]}", flush=True)
        if pod:
            s["pods"].append({"id": pod["id"], "ip": None, "port": None, "arms": None})
            save(s)
    # wait for ssh on all
    for p in s["pods"]:
        if p["ip"]:
            continue
        for _ in range(45):
            info = runpod.get_pod(p["id"]) or {}
            ports = ((info.get("runtime") or {}).get("ports") or [])
            up = [x for x in ports if x.get("privatePort") == 22 and x.get("isIpPublic")]
            if up:
                p["ip"], p["port"] = up[0]["ip"], up[0]["publicPort"]
                print(f"  {p['id']} ssh ready {p['ip']}:{p['port']}", flush=True)
                save(s)
                break
            time.sleep(10)
    save(s)


def bootstrap():
    s = load()
    for p in s["pods"]:
        if not p["ip"] or p.get("ready"):
            continue
        print(f"bootstrapping {p['id']} ...", flush=True)
        for src, dst in ((SAP, "/root/sap.tar"), (BOOT, "/root/bootstrap.sh")):
            subprocess.run(["scp", *SSHOPT, "-P", str(p["port"]), "-q", src,
                            f"root@{p['ip']}:{dst}"], check=False, timeout=600)
        r = ssh(p, "bash /root/bootstrap.sh 2>&1 | tail -4", timeout=1800)
        print("   ", r.stdout.strip().replace("\n", " | "), flush=True)
        p["ready"] = "BOOTSTRAP_OK" in r.stdout
        save(s)


def launch():
    s = load()
    ready = [p for p in s["pods"] if p.get("ready")]
    for p, arms in zip(ready, ARM_SETS):
        jobs = "".join(f"{m} {sd}\n" for m in arms for sd in range(1, SEEDS + 1))
        p["arms"] = arms
        cmd = (f"cat > /root/jobs.txt <<'JOBS'\n{jobs}JOBS\n"
               "rm -f /root/res/*.csv /root/sweep.done; "
               "nohup bash -c 'xargs -a /root/jobs.txt -n2 -P 94 /root/run_arm.sh "
               "> /root/sweep.log 2>&1; echo DONE > /root/sweep.done' >/dev/null 2>&1 & "
               "sleep 2; wc -l < /root/jobs.txt")
        r = ssh(p, cmd)
        print(f"{p['id']} arms={arms} jobs={r.stdout.strip()}", flush=True)
        save(s)


def status():
    s = load()
    tot = 0
    for p in s["pods"]:
        if not p.get("ip"):
            continue
        r = ssh(p, "cat /root/res/*.csv 2>/dev/null | wc -l; ls /root/sweep.done 2>/dev/null | wc -l", timeout=120)
        out = r.stdout.split()
        done, fin = (int(out[0]), int(out[1])) if len(out) >= 2 else (0, 0)
        tot += done
        print(f"  {p['id']} arms={p.get('arms')} rows={done} {'DONE' if fin else 'running'}")
    print(f"  TOTAL rows: {tot}")


def collect():
    s = load()
    dest = pathlib.Path("C:/Users/nikol/AppData/Local/Temp/sweepres")
    dest.mkdir(exist_ok=True)
    for p in s["pods"]:
        if not p.get("ip"):
            continue
        subprocess.run(["scp", *SSHOPT, "-P", str(p["port"]), "-q",
                        f"root@{p['ip']}:/root/res/*.csv", str(dest)], check=False, timeout=900)
    print("collected into", dest, "->", len(list(dest.glob("*.csv"))), "files")


def kill():
    s = load()
    for p in s["pods"]:
        try:
            runpod.terminate_pod(p["id"])
            print("terminated", p["id"])
        except Exception as ex:
            print("FAILED to terminate", p["id"], ex)
    STATE.write_text(json.dumps({"pods": []}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "provision":
        provision(int(sys.argv[2])); bootstrap()
    elif cmd == "bootstrap":
        bootstrap()
    elif cmd == "launch":
        launch()
    elif cmd == "status":
        status()
    elif cmd == "collect":
        collect()
    elif cmd == "kill":
        kill()
