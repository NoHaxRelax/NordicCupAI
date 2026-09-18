# Survival replay debugger

Open the running viewer at [http://127.0.0.1:9053](http://127.0.0.1:9053).

The viewer plays recorded simulations. Pause, scrub, step through frames, change speed, click a creature for stats, follow it around the map, and inspect its last policy decision. The live recording picker automatically discovers completed replays throughout the survival workspace, including every strategy result folder. Recordings are ordered by their `meta.created_at` timestamp with the newest first, while initial loading prefers the newest organizer-rendered replay. It refreshes every ten seconds while visible, on focus, and through **Refresh list**, without interrupting the current replay. Duplicate copies of the same recording share one option; separate runs and versions remain separate. Labels identify original simulator rendering (`native`) or the explicit lightweight state fallback (`state`).

Recordings capture the simulator's own `Environment.draw` output by default and open in **Simulator rendering** mode. That is the organizer renderer, including its hearing circles and obstacle-clipped vision cones. Untick the option to inspect the lightweight debugging view. Explicit state-only recordings use the fallback view, which shows nominal hearing and vision for every creature but cannot recover the organizer pixels after the run.

## Open it again

From the project root:

```sh
survival/.venv/bin/python survival/debugger/serve.py --open
```

Then open [the local debugger](http://127.0.0.1:9053). The server binds only to your computer. It serves the viewer plus validated replay files under `survival/`; it does not expose arbitrary filesystem paths. A background scan runs every five seconds, with changed files validated once and cached. Initial discovery can take longer for a large library. Stop it with Ctrl-C. No cloud account, deployment or running simulation is needed to play a recording.

The included [Survival Lab.html](<Survival Lab.html>) is an offline snapshot of up to 10 newest organizer-rendered recordings at build time. Open it directly in a modern browser and import other recordings through the file picker. Use the localhost viewer for all current runs. A complete snapshot including state-only recordings can be built with `build_portable.py --all`, but the accumulated recordings can make that export very large; future recordings require another export.

## Record a future run

The existing local Python environment already has the simulator dependencies. To install on another computer, create a virtual environment and install `survival/vendor/survival-simulator/requirements.txt`.

```sh
survival/.venv/bin/python survival/debugger/record.py \
  --mode nursery --seed 7 --seconds 3000 --every 1 \
  --output survival/results/nursery/replays/nursery-seed7-v1.json.gz
```

`--every 1` keeps each 0.1-second tick. Use `--every 5` for a smaller recording with regular frames every 0.5 seconds. Events are still tracked every tick. Births, deaths and predator state changes also preserve an exact frame between regular samples. The default duration is the complete 3000-second game or species extinction. Existing files are protected; use a different name or explicitly pass `--overwrite`.

`--mode` accepts `dummy`, `greedy`, `nursery`, or `speed`. For the arranged wall experiment, use `--scenario wall --seconds 70`. Every completed `.json` or `.json.gz` replay under `survival/` is discovered automatically, wherever a strategy saved it. Prefer `survival/results/<strategy>/replays/<unique-run-id>.json.gz`. `--register` remains a compatibility flag and is no longer necessary. Dependency/vendor folders and debugger test fixtures are excluded. Do not edit a manifest or copy finished files by hand. Imported files outside the workspace remain available through drag and drop.

Native organizer rendering is the default. `--native-render` remains accepted for explicitness; use `--state-render` only when a smaller state-only diagnostic recording is intentional. For example:

```sh
survival/.venv/bin/python survival/debugger/record.py \
  --mode nursery --seed 7 --seconds 120 --every 5 --native-render \
  --output survival/debugger/recordings/nursery-native-seed7.json.gz --register
```

Native images default to 960 pixels wide; use `--native-width 1600` for full world resolution. The original renderer draws the whole scene, including its own hearing/vision overlays, into each image. Those baked overlays cannot be toggled individually; use the lightweight view for separate switches. Native recordings are larger and slower to produce, so start with a short run or a larger capture interval. Generated maps preserve their original textures. The artificial wall fixture has no generated texture, so its terrain uses a flat color from the simulator's biome palette while all entities and obstacles still use the native renderer.

Only a replay with the supported schema can be imported. Previous score-summary JSON files do not contain positions or snapshots and cannot be played retrospectively; re-run the policy with recording enabled. This viewer does not ingest a video or an arbitrary simulator's log.

## Connect your team's controller

Provide a Python file containing a factory that accepts `seed=` and returns a callable. The callable receives `(agent_states, sim_time)` and returns a list of `(agent_id, ActionRequest)` pairs. These are the same ordinary observation dictionaries used by the local benchmark; the custom policy receives no hidden world state from this wrapper.

```sh
survival/.venv/bin/python survival/debugger/record.py \
  --policy /absolute/path/to/team_policy.py:make_policy --seed 7 \
  --output survival/debugger/recordings/team-seed7.json.gz --register
```

For readable annotations, the callable can expose `last_decisions` as `{agent_id: {"rule": "Collect fruit", "detail": "Nearest safe fruit selected."}}`. It should set this after deciding each batch of actions. Use short descriptions of policy rules and targets. Plain action values are recorded even without annotations.

For an existing simulation loop, the recorder can also be imported directly. Add `survival/debugger` to your Python path:

```python
from recorder import ReplayRecorder

recorder = ReplayRecorder(sim.env, title="Team policy, seed 7", seed=7, every=1)
recorder.capture(force=True)
state = sim.step([])
recorder.capture()
while state["num_agents"] and state["sim_time"] < 3000:
    inputs = state["observations"]
    action_t = state["sim_time"]
    actions = policy(inputs, action_t)
    state = sim.step(actions)
    recorder.capture(actions, getattr(policy, "last_decisions", {}), inputs, action_t)
recorder.save("survival/results/team/replays/team-seed7-v1.json.gz", reason="game finished")
```

Record every run, including controls and failures. Use distinct filenames for new versions and trials. Call `save` only after the final step; it publishes a fully closed file atomically, so partial recordings never enter the catalog. A failed save preserves an existing completed file. See [the run requirements](../AGENTS.md).

Create a new recorder for each environment/reset. It reads entity state, does not call the simulator RNG, and does not modify gameplay state. Native captures use isolated render layers and restore creature vision caches afterward. Integration tests verify state/RNG preservation and pixel equality with `Environment.draw`. Replays describe the actual recorded run, rather than reconstructing behavior from a seed alone. Direct integrations can pass `native_render=True` to `ReplayRecorder`.

## What the debugger knows

The main map and inspector are an omniscient debugging view. They include true positions, predator energy/rest status, fruit maturity, and hidden agent old-age thresholds. These are **not** all available to a competition policy. The inspector also contains the ordinary engine observations and the inputs that produced the last action.

Frames show post-step world state. The last action and its decision annotation refer to the preceding input, identified by `action_t`. The upstream engine itself caches observations before predators move, so the reported predator distance can differ from its displayed current position. Vision/hearing overlays are range geometry, not a replacement for the exact observed-object list or wall occlusion checks.

Deaths are detected from membership changes and attributed to the last known position; their cause is not guessed. Fruit-removal events mean eaten **or** rotted, with no invented eater. Tree/predator IDs are stable debugger IDs; the player interface does not provide them. Fruit internal age advances twice as fast as simulated seconds.

## Checks

```sh
survival/.venv/bin/python -m unittest discover -s survival/debugger/tests -v
```

Replay format is versioned as `format: "survival-replay", version: 1`. The JavaScript viewer requires a modern browser supporting Canvas and `DecompressionStream` for gzip imports. Replays and stats stay in the browser; no external requests are required by the portable viewer.

The local HTTP viewer has been browser-tested. Direct `file://` testing of the portable HTML was blocked by the browser automation URL policy, so that route is not claimed as verified.

Inspect the full live catalog without starting the server:

```sh
survival/.venv/bin/python survival/debugger/catalog.py
```

The live catalog endpoint is `http://127.0.0.1:9053/recordings/manifest.json`. It reports source paths, aliases, counts and incomplete/invalid files. Score-only JSON is not a replay; it must be reproduced with recording enabled, never converted into invented footage.

Rebuild the portable snapshot of up to 10 newest organizer-rendered recordings after changing viewer code (add `--all` to include the complete state-only library):

```sh
survival/.venv/bin/python survival/debugger/build_portable.py
```
