# Seed-aware C++ survival policy research

Selected development policy: **mode 46**, average **1809.62** on 128 fresh paired confirmation seeds, versus 1725.75 for mode 8. Paired gain 83.87, bootstrap 95% interval [29.65, 140.18], 76 wins and 52 losses. **The 2200 target is not achieved.**

The controller retains Lucas orchard behaviour and full current resources and map. After configurable model activation, currently 180 simulated seconds, proposed births are checked against the synchronized model RNG. Mode 46 postpones a birth until the forecast child has an ageing threshold of at least 90, walking speed at least max(parent speed, 12), and nondecreasing sprint speed. Parent energy below 115 bypasses the delay. Only accepted births advance the forecast RNG; live RNG is untouched. Every accepted forecast is checked against the actual offspring. All 46,847 birth forecasts in confirmation matched, and all 128 paired pre-activation hashes matched.

This requires a fully synchronized model including RNG, not merely a seed number or map. It is a native experiment with access to exact current state, not unknown-live validation or a promise of immutable future spawns. No evaluation or validation API client is included.

Use Linux Python 3.12 and NumPy 2.3.5. Python initializes NumPy compiled math kernels once; the simulation and policy loops are C++. Pin PYTHONPATH to that environment for the embedded interpreter. Build with `python build.py`, then run `bench/policy_bench configs/orchard.json SEED 46 3000 OUTPUT 180`.

Rejected experiments are retained: exact predator location mode 27 failed 128-seed fresh confirmation; one-step predator action search modes 35 to 37 failed 16 paired seeds; exact-age lifecycle modes 39 to 42 failed 16 pairs; early fruit arrival modes 49 and 51 failed 16 pairs. Mode 50 inventory had a small unconfirmed gain. Their presence in history is not promotion. The next experiments are outside this frozen checkpoint.

Canonical numerical portability was verified by a full matching baseline game on mypc and Runpod with NumPy 2.3.5. Older NumPy 1.26.4 results are historical exploratory evidence and must not be pooled. Full completed state-only replays are retained locally; GitHub stores compact source and result evidence.
