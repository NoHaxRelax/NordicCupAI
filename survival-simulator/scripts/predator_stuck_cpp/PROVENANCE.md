`vendor_engine.cpp` derives from `survival-simulator/fastsim/_engine.cpp`
in the sibling NordicCupAI checkout, captured 2026-09-19.

Original uninstrumented SHA-256: `45a5928464b5231feb8b1378fc0c4b593f22c7ba38040274ae7f4295d2581f5f`.

The upstream header identifies the original simulator source commit as `acfc31a`.
This copy contains the existing port's CPython random/set semantics, map RNG
draws, GEOS-compatible geometry, and NumPy float64 C loops. `_stuck.cpp` adds
the bulk loop and confinement detector. The vendor copy now also records passive
movement telemetry: pre-move state, selected edge, decision branch, attempted
directions and rejected/accepted collision candidates. Instrumentation neither
consumes RNG nor changes decisions. `verify.py` checks instrumented bulk runs
against the unmodified Python engine and replays logged accepted moves.

The native math uses NumPy C function pointers, with no Python calls per tick.
NumPy is deliberately retained to match its CPU-specific numerical behavior.
The source does not promise identical trajectories across operating systems or
NumPy builds. Run `verify.py` on each target platform. The original port uses
creation-order object hashes instead of Python address hashes; agent-free
predator/RNG parity is checked directly against unmodified Python objects.
