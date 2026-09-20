# Full fast1m endpoint checkpoint

Imported from the running `Val checking` deployment at
`/opt/nordiccup-fast1m`, frozen strategy source `8721a3d`.
Strategy changes are confined to HTTP route aliases, per-service log paths,
and exposing settings on the health response. Native policy and harvester
are copied unchanged.

Configuration: enabled, budget 1,000,000, maximum 100 harvests, sacrifice mode
`predict_contact`, cooldown 0, contact margin 1.0, minimum free agents 6,
random tail disabled, no score cap. This uses the duplicate-action game bug.
It does not recover seeds or deliberately stall responses.

Both former seed endpoints run this strategy independently:

- Hetzner: `http://46.62.244.29:9064/seed-live-mode144/predict`
- Runpod: `https://ft7k34t881e55j-19123.proxy.runpod.net/seed-live-eval-20260920/predict`

`/health` and either former `/status` alias report the strategy and parameters.
The original `/predict` route is also supported. No new queue submission is
needed: whichever address the queued attempt retained receives fast1m.

Hetzner uses its existing Python 3.14 / NumPy 2.3.5 native binary. Runpod builds
the same C++ source for Python 3.12 / NumPy 2.3.5. Both use FastAPI 0.121.2 and
Uvicorn 0.38.0. The ten contact-predictor regression checks passed on both hosts.
That does not establish full-game numerical parity between Python versions.

Hetzner service: `fast1m-9064`; separate logs `/var/log/fast1m-9064.jsonl`.
Runpod source: `/workspace/fast1m-endpoint`; logs
`/workspace/fast1m-19123.jsonl` and `/workspace/fast1m-19123.log`.
Seed services/search workers are stopped; pods remain allocated.
