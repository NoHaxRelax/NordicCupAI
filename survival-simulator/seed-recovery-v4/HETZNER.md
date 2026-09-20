# Hetzner live seed controller

Deployed at `http://46.62.244.29:9064/seed-live-mode144/predict`.
Service: `hetzner-seed-live.service`; source/runtime: `/opt/seed-live`.
The existing services on 9052, 9054 and 9062 were not replaced.

- Native C++ build on Hetzner, Python 3.12.14 and NumPy 2.3.5.
- Seed spoof fast64 scanner; full uint32 range; seven filter workers and one
  wall verifier. Search processes stop after candidate recovery.
- Public orchard actions during search; response pacing capped at nine
  seconds with a 540-second cumulative extra-wait budget.
- Shadow engine replays actual actions and supplies Oscar mode 144 actions
  after public-state verification. Empty bootstrap packets return no actions.
- Current pending frames are compared before shadow control; historical
  catch-up checks and detailed receipts are sampled every 100 ticks.

The final bounded smoke test recovered seed 123456789 in 8.3598 seconds at
simulation time 75.3, stopped its search process group, and served model actions
without a recorded mismatch. This is an early-range seed, not a full-domain
runtime measurement or a completed game score. Receipts: `evidence/hetzner`.

The validation enqueue POST with the direct Hetzner URL returned the existing
attempt `5d05fcaba9a54b00a74c3358267351de`, position 21. The API exposes no queued
URL, cancellation or editing operation, so replacement of the stored URL is
unconfirmed. The former Runpod service is stopped. Its callback address now
runs `hetzner_queue_bridge.py`, forwarding requests to Hetzner if the queue still
uses the old URL. That fallback retains the network hop and cannot establish
direct-Hetzner latency. Inspect the eventual attempt's `service_url` to tell.
No additional Runpod pod was rented or terminated for this migration.

Check service: `ssh root@46.62.244.29 systemctl status hetzner-seed-live`

Check progress: `curl -s http://46.62.244.29:9064/seed-live-mode144/status`
