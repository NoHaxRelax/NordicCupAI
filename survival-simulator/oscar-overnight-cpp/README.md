# nightsim: fast C++ survival simulator + policy (Oscar's overnight branch)

`nightsim/` is a private fork of the native survival engine (bit-exact C++ port of the survival simulator) with the
orchard policy and all predator layers (evasion, guides, crevice traps, refuge, map fixes) in C++, behind Python
bindings. Nothing here talks to the competition API.

## Speed (measured 19 Sept)
| Where | Time per full game | Throughput |
| --- | --- | --- |
| Laptop, one process | ~3 s | only for build checks |
| Runpod cpu3c pod, 32 vCPU, 32 games at once | ~29 s per game per core | ~3,600-4,300 full games per hour per pod |
| 8 such pods | | ~29,000-34,000 full games per hour (~$7.70/h) |

Games are single-threaded: run one worker per core. Guide/delivery scenarios take well under 1 s each.

## Build and run (any Linux or macOS box with Python 3.12)
```bash
pip install numpy pydantic
python nightsim/build.py
python nightsim/run.py --configs configs/best-configs.json --seeds 1-64 --predators --workers 32 --out rows.jsonl
python ana.py rows.jsonl --ref pred_best_0400
```
`--configs` takes a JSON file or inline JSON `{"label": {params...}}`; every policy parameter is listed in
`nightsim/_nengine.cpp` (`parse_params`). Best configs: `configs/best-configs.json` (`r21s0c2` without predators,
`pred_best_0400` with predators); `configs/configs-all.json` also has `guide_best_1119` (delivery guide).

## On Runpod (the fast way)
1. Create a CPU pod: cpu3c, 32 vCPU, image `runpod/base:1.4.0-rc.164-ubuntu2404`, port 22/tcp, your SSH public key.
2. Add a line to `pods.txt`: `name host port pod_id /workspace/night/.venv` (see `pods.example.txt`).
3. `./night.sh bootstrap NAME` (venv + numpy), `./night.sh deploy NAME` (copies nightsim/ and builds),
   `./night.sh launch NAME JOB "--configs /workspace/night/cfg.json --seeds 1-96 --predators --workers 32 --out /workspace/night/runs/JOB.jsonl"`,
   `./night.sh pull NAME JOB`, `./night.sh status`.
   Copy config files up first: `cat cfg.json | ./night.sh ssh NAME "cat > /workspace/night/cfg.json"`.
4. Container disks are wiped when a pod stops: bootstrap + deploy again after every restart. Stop idle pods.
Set `NIGHT_SSH_KEY` if your key is not `~/.ssh/id_ed25519`.

## Scenario harnesses (narrow tests)
- `nightsim/guide.py`: guide delivers one predator to a crevice with a bait. Env: `NIGHT_GE` guide energy,
  `NIGHT_NPRED` predators, `NIGHT_NBY` nearby colony agents, `NIGHT_NOLINE=1` allow blocked lanes,
  `NIGHT_TRACE=1` per-tick trace (one job, `--workers 1`). Analyze with `edge_ana.py` / `g_ana.py`.
- `nightsim/hold.py`: a colony keeps a bait alive while a predator is held at the mouth (600 s).
- `nightsim/refuge.py`, `escape.py`, `trapsite.py`, `mapcheck.py` (map accuracy against the true walls).
- Full games with per-death records: `NIGHT_DEATHS=1 run.py ...`, analyze with `diag_ana.py`.

## Notes
`notes/STATUS.md` (chronological log and morning summary), `results/results-summary.md` (all result tables),
`notes/BEST.md`.
