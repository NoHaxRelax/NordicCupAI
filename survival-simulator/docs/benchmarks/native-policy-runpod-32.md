# Native C++ policy throughput

Measured 2026-09-19 on one Runpod Secure Cloud `cpu3c` instance with 32 vCPUs. Thirty-two games ran concurrently with one process per core. Predators were enabled and the horizon was 3000 simulated seconds.

- Batch wall time: **36.161 seconds**
- Throughput: **53.10 games/minute**, or **3,186 games/hour**
- Simulated game length: median **1332.7 s**, mean **1396.8 s**, range **747.5–2496.6 s**
- Per-game wall time: median **15.933 s**, mean **16.900 s**, p10 **9.962 s**, p90 **25.169 s**, range **6.183–36.058 s**

The earlier Python-policy pilot took 52.016 seconds for 253.2 simulated seconds. Normalized by simulated time, the native median is about **17 times faster**. The comparison is directional because the seeds and game lengths differ.

