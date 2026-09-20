# Rust seed search on Runpod

The Runpod benchmark completed on 19 September 2026 using **all nine allocated CPU threads**. Three full ten-million-seed scans took **8.57, 8.63 and 8.76 seconds**, a median of **8.63 seconds** or approximately **1.16 million seeds/second**. Every scan returned only seed **78431**.

A newly collected Linux survey also recovered seed 78431 from ten million candidates and matched **all 300 public frames** during native replay. Its search plus verification took **12.52 seconds**, after collection. The 30-simulated-second collection took **6.56 seconds** of accelerated local wall time on the pod.

## Allocation and CPU use

- Pod: `37ntp2j4x856ds`, Secure Cloud, `CA-MTL-1`.
- Host CPU: Intel Xeon Gold 6342 at 2.80 GHz.
- Allocation: an A40 pod with **9 vCPUs**, listed at **$0.49/hour**. The Rust implementation used the CPUs; it did not use the GPU.
- The host exposed 96 logical CPUs. The benchmark explicitly used nine worker threads, respecting the pod's allocation.
- Linux cgroup v1 quota: **765,000 microseconds per 100,000-microsecond period**, equivalent to **7.65 fully occupied CPU cores**.
- Measured average CPU usage in the three scans: **7.58, 7.61 and 7.55 cores**, approximately **98.7–99.5% of the available CPU quota**. This confirms that the search used the allocated compute capacity.
- Rust 1.98.1, release profile, pinned Cargo dependencies; CPython 3.12.12 for survey/replay.

CPU-only allocations for 32, 16 and 8 vCPUs were rejected by Runpod with an unavailable-instance error, through both the connector and a direct API request. An A40 allocation requiring 32 CPU threads was also rejected. The ordinary A40 allocation succeeded with nine CPU threads. Existing simulation pods were not stopped or changed for this benchmark.

The local machine's earlier 20-thread result was 6.96 seconds for the same saved Windows signature. This nine-thread cloud allocation was therefore slower for the prefix scan; renting a pod alone does not guarantee a speedup.

## Measurements

All seed ranges begin at zero. Ten million candidates means every integer through 9,999,999 was checked.

| Input | Candidate count | Search threads | Prefix search time | Survivors |
| --- | ---: | ---: | ---: | --- |
| Saved Windows survey, single-thread reference | 1,000,000 | 1 | 5.55 s | 78431 |
| Saved Windows survey, repetition 1 | 10,000,000 | 9 | 8.57 s | 78431 |
| Saved Windows survey, repetition 2 | 10,000,000 | 9 | 8.63 s | 78431 |
| Saved Windows survey, repetition 3 | 10,000,000 | 9 | 8.76 s | 78431 |
| Fresh Linux survey | 10,000,000 | 9 | 8.60 s | 78431 |

The single-thread measurement corresponds to about 180,076 seeds/second. The repeated nine-thread median gives approximately 6.44 times that throughput. Counts differ between those measurements, so this is a throughput comparison rather than a same-size paired timing.

Seven Rust compatibility tests and five saved-survey integration tests passed on the pod before timing. Upload hashes were verified for all 60 selected source/configuration/data files. The source and input hashes in the downloaded results were checked locally before termination.

## Native verification and platform boundary

The saved Windows survey's biome filter produced the same candidate on Linux, and native **founder headings and rock dimensions matched**. However, its exact public-frame hashes diverged at frame zero on Linux. This run does not claim bitwise replay portability between Windows and Linux. The underlying cause of that hash difference was not isolated here; floating-point/runtime differences are a possibility.

To test full recovery on the actual remote runtime, the benchmark collected a **separate 30-second Linux survey**, using the shared world estimator and public observations. That survey retained 64 selected biome samples. It left one candidate in the ten-million range, and fresh native replay matched **all 300 recorded Linux frames**.

The Linux search took 8.60 seconds and native verification 3.90 seconds, with total runner wall time 12.52 seconds. This last measurement used the already initialized Python benchmark process and excludes environment setup, compilation, upload/download, and observation collection. The current policy source/configuration snapshot is recorded with the Linux survey; it should not be assumed identical to the earlier Windows survey's effective settings.

## Evidence and reproduction

- [Full benchmark JSON](seed_search_runpod_benchmark_2026-09-19.json): individual scan timings, measured child CPU time, source/binary/input hashes, both verification outcomes.
- [Host and cgroup evidence](seed_search_runpod_host_2026-09-19.json).
- [Fresh Linux survey and action log](seed_search_runpod_linux_survey_2026-09-19.json).
- [Cleanup receipt](seed_search_runpod_cleanup_2026-09-19.json).
- [Remote benchmark helper](../scripts/seed_search_rust/tests/runpod_benchmark.py) and [Linux setup/run script](../scripts/seed_search_rust/runpod_run.sh).
- Full downloaded logs and archive are retained locally under `runs/seed-search-runpod-20260919/` (ignored by Git). Archive SHA-256: `b6d4147a4ba35298507c1224040e509ad01ba4a59b9512a7888b3c6c94bece3e`.

On a future Linux pod, copy an isolated `survival-simulator` source tree, including the Rust crate, required simulator modules/configs, seed helper scripts, and the two saved survey JSON fixtures. From that directory, set the allocation reported by Runpod and run:

```bash
SEED_SEARCH_THREADS=9 SEED_SEARCH_POD_ID=YOUR_POD_ID bash scripts/seed_search_rust/runpod_run.sh
```

The launcher installs the pinned Rust/Python dependencies, runs compatibility checks, and writes results to `runs/seed-search-runpod/`. Use the pod's allocated vCPU count rather than the full host count reported by `nproc`. Infrastructure provisioning and termination remain the caller's responsibility; the shell script itself does not manage billing.

The temporary pod was deleted after download validation: the delete returned HTTP 204, and the subsequent lookup returned HTTP 404. The independent 30-minute cleanup guard also confirmed it was already deleted. Billing records for this pod had not yet appeared at the final lookup; the hourly allocation price is not a final invoice.

This run used the [Runpod skill](C:/Users/nikol/.codex/plugins/cache/openai-curated-remote/runpod/1.1.2/skills/runpod/SKILL.md), its MCP control-plane tools and SSH for execution and transfer.
