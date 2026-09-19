# Short-iteration baseline — 19 September 2026

Production behavior remains the default guide and coordinator at commit
`53f1a4c`; later changes add debug output and reports. Save this checkpoint
before user-directed replay iterations. No experimental policy has silently
replaced the measured default.

| Protocol | Result |
| --- | --- |
| Default 30+1 delivery, 100 maps | 74/100 |
| Offline-selected sites, independent encounters | 319/380; 19/20 maps empirically above 50% |
| Predictive pacing candidate, reused development maps | 78/100 versus 74/100 |
| Static ranking candidate, fresh 100 maps × 3 encounters | 219/300 versus 213/300 |
| Generic corners on 7 previously unsupported maps | 32/42; replacement survived 42/42 |
| Default native games, 10 complete recorded games | Mean score 521.46; all eventually extinct |
| Updated orchard integration, 6 native games | Mean 814.93 versus 585.50 on same 6 baseline seeds |

The six-game comparison is small and native runs are not bit-identical across
processes. Isolated trap results assume preloaded predators, known static
walls, full initial guide energy and permanent bait. Real games use normal
energy, exploration and reproduction. Do not combine these protocols into
a single reliability percentage.

The 95% target was not met. Active iteration stopped overnight, although
submitted jobs completed. New work is one C++ real-game replay followed by
user feedback.
