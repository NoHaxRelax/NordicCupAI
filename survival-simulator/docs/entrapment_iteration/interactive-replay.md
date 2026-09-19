# Interactive C++ replay loop

Baseline checkpoint: `entrapment-interactive-baseline-20260919` (`3a2391e`).
Experimental candidates are preserved separately under
`entrapment-research-candidates-20260919`; they are not the default policy.

The first interactive replay uses seed 1883894846, ordinary energy, native
predator spawning, and observation-only policy inputs. Requested horizon
3000 seconds; natural extinction at 626.8 seconds. Score 632.18. All 6,268 frames
are saved locally under `logs/entrapment-iteration/fast-recorded-seed1883894846`.
This is one inspection game, not an estimate of mean performance.

Run from repository root (current machine's pinned C++ checkout):

```sh
.venv/bin/python survival-simulator/scripts/fast_entrapment_game.py --fastsim /home/Ucals/projects/NordicCupAI-fastsim-inspect/survival-simulator --seed 1883894846 --seconds 3000 --out survival-simulator/logs/entrapment-iteration/NEW-RUN
.venv/bin/python survival-simulator/scripts/entrapment_viewer.py survival-simulator/logs/entrapment-iteration/NEW-RUN --port 9068
```

Use a new output directory for every iteration. The C++ checkout is pinned
to `696bbd86272c27d9edba56238d3ec3b9469faaf9`; its compiled extension is
Python/platform-specific. Build with that checkout's fastsim/build.py when
needed. Recorded manifests preserve policy and engine source hashes.

The viewer must remain in a persistent foreground terminal session. A child
backgrounded from a transient shell exited and made the first link fail.
Reload the page after recording completes to obtain the full frame range.
