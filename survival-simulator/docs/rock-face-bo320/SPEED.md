# Integrated speed check

Policy changes from d6486e4 (including f31f586), engine changes from the frozen working snapshot in engine-snapshot.patch. The source tasks were “Speed up policy” and “Speed up engine?”. Engine work was still active when this snapshot was taken; these are independently checked changes, not a claim that the other task has finished all experiments.

Three full rock-face games matched every action: 51,098 ticks, seeds 25001/25008/25016. Short lockstep checks against Python also passed with 30 starting predators; this is finite evidence, not a proof for all maps. The policy boundary checker passed on all ten pods.

Same-pod 32-worker sample on seeds 25001–25032: every game matched score, lifetime, tick count, peak population and death counts exactly. Mean native-loop wall time was 18.15 seconds before and 11.82 after. Mean process CPU time fell from 11.88 to 9.96 seconds, a 16.2% improvement. Different scheduling delays account for part of the larger wall-time reduction; do not attribute the entire wall improvement to code. Timing excludes initialization. No search parameters were selected from this pilot.

Reproduce and inspect using scripts/check_rock320_speed.py and scripts/pilot_rock320.py. Artifacts and exact source/build provenance are stored alongside this note.
