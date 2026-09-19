# Nikolaj rank1 comparison
User-requested independent evaluation of commit e8c8628862066d3e59226a3d35bd365b635be81d, branch sim-optimization-no-trapping.
Configuration: models/best_policies/rank1_pod-03.json, recommended by that commit's BEST_POLICIES.md. Newer untracked study winners are not included in the commit; clarification requested.
Run unchanged native policy and engine from that commit. Map seeds 19001–20000, policy seed 0, natural predators, horizon 3000 seconds, profiling enabled, no initial empty step (same harness convention as sharedfood20).
Host: ssh pc, Ryzen 5 3600 (Zen2), 12 logical CPUs. Directory /home/lucas/nikolaj-e8c86288-eval1000/survival-simulator. Interpreter /home/lucas/entrapment-research-20260918/.venv/bin/python.
Results appended each game and fsynced every 10, in results/nikolaj1000. Build provenance, hardware, configuration hash saved in manifest. eval.log reports progress.
Cross-hardware floating-point differences documented upstream mean matching seeds do not guarantee matching world trajectories against the Runpod panel. Report this limitation and do not attribute hardware differences to policy quality. No official validation submission.
