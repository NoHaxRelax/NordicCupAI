# Survival simulation runs and replays

This is local hackathon research. Do not call competition validation/evaluation APIs or submit scores without explicit authorization.

## Every completed simulation run must be inspectable

- Record every completed simulation run, including controls, failures, and parameter-sweep cases. Saving only aggregate scores or a representative success does not satisfy completion.
- Use `survival/debugger/recorder.py` and its `ReplayRecorder`. Capture the initial state, call `capture` after every simulation step, and call `save` at the end. Its sampling interval may reduce stored frames; births, deaths and other critical events retain exact frames.
- Preferred location: `survival/results/<strategy>/replays/<unique-run-id>.json.gz`. Include policy/version, seed and a unique suffix in the filename. Preserve completed runs rather than overwriting them with the next version.
- Include an informative title, policy, seed, scenario and limitations in recorder metadata. Mark arranged geometry, hidden-state access, prepared food, and shortened horizons accurately.
- Use the original simulator rendering (`native_render=True`) for demonstrations intended for close visual inspection. Lightweight state recordings are acceptable for large sweeps; do not omit the replay to save native-rendering cost.
- `ReplayRecorder.save` publishes files atomically. Survival Lab automatically discovers complete `survival-replay` JSON/gzip files anywhere under `survival/`, excluding dependency/vendor directories and debugger test fixtures. No manifest editing, manual copy or `--register` flag is required.
- A finished run should appear in the live recording picker after the server scan and browser refresh (normally within about 15 seconds once the initial scan is complete). The user can also click **Refresh list**.
- Check that your completed run appears at `http://127.0.0.1:9053/recordings/manifest.json` before calling the work delivered. If the server is stopped, run `survival/.venv/bin/python survival/debugger/catalog.py` to verify discovery.
- Old score/summary files without recorded world states cannot be reconstructed as the original replay. If backfilling, rerun the saved controller/configuration locally and label the recording as a new reproduction. Never fabricate missing frames or claim rerun footage is the original run.
- Do not modify the vendored simulator to record a run. The recorder must remain read-only with respect to simulation state and RNG.

The live viewer is the current complete library. `Survival Lab.html` is an offline export snapshot, which must be rebuilt with `build_portable.py --all` to include the complete current library.
