# Active burst score pilot

Branch codex/burst-score; source baseline 43e3d52.
Eight variants, 68 shared maps, 544 games. Local simulations only. No hosted validation or evaluation calls.
SSH PC: seeds 91001–91004 (32 games), ~/lucas-burst-score/pilot-pc.
Runpod shards: 91005–91020, 91021–91036, 91037–91052, 91053–91068 (128 games each).
Each pod runs /workspace/burst-score/pilot with log /workspace/burst-score/pilot.log.
Pod IDs, created in this conversation: 4j2tnn9asu05qx, 8l5ytgcntiy8b5, w1fraeeejt1djk, f6stnjjkun6vhu.
Price $1.12/hour each, $4.48/hour total plus storage.
User requests closing these pods after results are retrieved.
No automation or scheduled follow-up was created. User will notify this thread on completion.
Progress: bash survival-simulator/scripts/burst_score_progress.sh
Collect raw games and manifests, then delete these four pods; do not touch other pods.

COMPLETED: all 544 games retrieved and verified. All four Runpod pods deleted with successful 204 responses. Results: RESULTS.md. No jobs remain active in this pilot.
