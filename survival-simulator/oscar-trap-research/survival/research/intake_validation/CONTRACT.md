# Intake validation contract

User objective: observation-only accumulation of33 predators, retained through
a3000-second game. Guide sacrifices and unlimited agent food are allowed.

1. Admission must happen individually or in explicitly named successive batches;
   a single preassembled33-predator group is only a holding control.
2. Default sparse schedule:33 arrivals at t=0,90,...,2880; horizon3000. Also stress
   with12-second intervals. Include awake/resting and lateral/heading perturbations.
3. Policy accepts native observation DTOs and public time only. Never supply
   predator IDs, energy/rest flags, true geometry/positions, selected targets or
   evaluator feedback. Prepared crew/arrival sites must be disclosed as fixtures.
4. Track admission delay, maximum simultaneous retained, final retained, last
   30-second minimum, holder/guide deaths, and every physical escape after first
   acquisition. Distinguish temporary guide-target switches from physical escape.
5. Do not call an attempt with missing arrivals a33 success. Surviving baits and
   a high average holding fraction are insufficient.
6. Freeze candidate sources before held-out seeds; retain hashes and every replay.
   Use native rendering for at least one final inspectable demonstration.
7. Separate a local intake protocol from full-map scouting/guide recruitment.
   Do not infer arbitrary native-game success from prepared approaches.
8. Root monitors live usage. Stop on /tmp/predator-intake-stop or root instruction,
   preserve completed artifacts and label interrupted runs.
