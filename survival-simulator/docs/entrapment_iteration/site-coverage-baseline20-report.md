# Independent encounter validation of selected sites

Twenty randomly sampled maps; screen up to three ordinary sites and two corner
sites using six encounters each, then test the best on twenty fresh encounters.
The policy has ordinary predator observations and known static walls.

- 19/20 maps had a candidate.
- 319/380 validation attempts succeeded on those maps (83.9%).
- Counting the missing-site map as twenty failed attempts: 319/400 (79.8%).
- All 19 candidate maps exceeded 50% success in validation.
- 15/20 maps had a per-site Wilson 95% lower bound above 50%. These intervals
  are not simultaneous confidence bounds; twenty maps is a small map sample.

The native success gate requires newcomer delivery, continuous original-group
retention, the final 30-second group hold, and a clear replacement side.
This batch checks replacement-side clearance but does not execute an actual
bait replacement; that remains a separate validation requirement.
Offline screening selects the site using simulation outcomes. It measures
site potential, not the deployed agent's ability to choose a good site from
geometry alone. A separate fresh-100 paired test evaluates a static ranking.

The ranking batch's first launch was invalid: all 429 workers exited before
simulation because its frozen guide_multi lacked --site. After adding the
existing site-index CLI plumbing, a real nonzero-site native smoke passed.
The corrected batch uses identical seeds in a new output directory; failed
launch output remains preserved and excluded from policy statistics.
