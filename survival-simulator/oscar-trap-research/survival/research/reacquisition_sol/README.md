# Global reacquisition diagnostic

In v20 map 10142, the tracked predator briefly selected bait at 38.2 seconds
while still 236 units from the mouth. This is ordinary distant line-of-sight,
not capture. It repeatedly lost sight and wandered; the receipt's physical
entry at 273.5 seconds is the meaningful geometry event.

The guide lost sustained pursuit near the start and spent most of its life on
the inherited `bounded_last_observation_reacquisition` rule around one stale
point. `GlobalSweepGuide` retains v24 behavior for the first three seconds of a
loss, then visits the existing static-map serpentine search points. It uses
only the guide's DTO, public time, and static map. No predator truth enters the
search order or actions.

The first fitted sweep crossed the predator's path at 30.5 seconds while they
were 62 units apart, just outside hearing. Its arbitrary scan heading missed
the predator; cached-action latency then reduced separation to 41 and 20 units
before death at 30.8. `GlobalSweepGaze` changes only sweep orientation: after
each movement it faces along that segment, putting an unseen predator ahead in
the next vision cone.

The gaze variant passed the fitted map 10142 run. It delivered at 65.4 seconds;
the guide sacrifice occurred at 65.3 immediately beside intake. The predator
remained physically contained through the 300-second horizon. The active-bait
audit confirms capture from 65.4 with no loss and does not flag a distant
autonomous arrival. All 3,001 native frames verify. This is one fitted repair,
not a fresh reliability estimate.

Map 10139 is a different failure. From 6.5 through 7.6 seconds the guide held
roughly 35 units of physical separation on grassland/forest. At 7.7 it crossed
into river while the predator remained on faster terrain: separation fell to
29, then 17.3 at 7.8, and the guide died at 7.9. A reacquisition sweep does not
address this. A future route scorer must reject slow-terrain entry during close
pursuit unless its observation-based predator model preserves adequate margin;
no additional fitted run was made here.

Map 10136's v20 receipt is also a strict attribution failure despite its
success flag: the tracked predator last targeted the guide at 5.1 seconds and
did not target bait until 62.3, after long unobserved wandering. Replaying the
new `GlobalSweepGaze` stack (which inherits v24 site selection) produced a real
delivery at 13.0 seconds and stable containment through 300 seconds. The guide
sacrifice occurred at 12.8 beside intake, and the audit does not flag an
autonomous capture. However, the global-sweep rule was never exercised in this
run; v24 selected a different refuge and maintained pursuit. This is a useful
regression pass for the combined stack, not evidence that sweep reacquisition
repaired map 10136.
