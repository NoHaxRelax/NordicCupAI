# Fast guide counterfactuals

`scripts/replay_guide_forecast.py --public-input-out FILE` saves the ordinary
DTO, local observed edges, estimated bait and pre-search memory while rebuilding
an original failure. Geometry caches are omitted and reconstructed. The action
is the recorded output of the original steering, not its earlier unsteered request.
Use `--time SECONDS` to capture a guide's earlier approach instead of its fatal
tick; the guide must be assigned at that time.

Subsequent search variants do not need to rebuild the whole policy. From the
`survival-simulator` directory:

```bash
../.venv/bin/python scripts/guide_forecast_fixture.py \
  docs/entrapment_iteration/guide-border160-public-input.json \
  --search models/entrapment/guide_lookahead.py --out /tmp/guide-decision.json
```

This case takes about 0.2 seconds locally. Add `--replay RECORDING --fastsim ENGINE`
on the original recording's host to replay recorded prior actions in the native
engine and replace only the chosen guide's action. That verifies one native tick,
not future survival. It checks living-agent IDs, positions, time and the exact DTO
before substitution. Native world state never enters the decision function.

The saved guide-160 fixture reproduces the current guard's STOP, 63/81 predicted
capture scenarios and native death, with zero position error. This matches the
slower full-policy reconstruction. A tested minimal ranking change also chose
STOP and died. Even the ordinary avoidance module's sprint-20 action died on
this tick. None of those is a successful repair.

Artifacts: `guide-border160-fixture-guard-native.json`,
`guide-border160-ranking-counterfactual.json`,
`guide-border160-ranking-experiment.patch` (apply to `f6583fb`), and
`guide-border160-escape-counterfactual.json`. The older fixture retains inert
cache-type markers from its original capture for hash reproducibility; the
current capture tool omits those cache keys entirely.
