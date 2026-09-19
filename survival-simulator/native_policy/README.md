# Native policy

This is the C++ simulator and policy loop from Oscar's `survival-simulator/oscar-overnight-cpp` branch, brought into the main entrapment work so full games can be evaluated without crossing the Python boundary every tick. `_npolicy.hpp` contains exploration, orchard survival, predator evasion, crevice bait, replacement, and guide logic. `_nengine.cpp` contains the simulator and invokes the policy directly.

Build and run the timing-only benchmark:

```bash
python nightsim/build.py
PYTHONPATH=. python benchmark.py \
  --config configs/with-predators-best.json \
  --games 32 --workers 32 --out native-timing-32.json
```

The benchmark retains only each seed's simulated duration and wall duration. The 2026-09-19 Runpod `cpu3c` result is stored in `../docs/benchmarks/native-policy-runpod-32.json`.

