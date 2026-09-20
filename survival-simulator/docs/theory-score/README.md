# Observed-biome risk pricing

The policy previously valued tree posts by food utility and distance. The winning change also prices the biome at each post using the agents' shared observed map. Slow biomes receive a larger penalty because they increase travel energy and exposure time. This uses only information observed by agents.

A 200-map paired pilot selected two fixed candidates. A fresh 2,000-map paired evaluation then compared them with the unchanged expanded-local-food baseline. `expanded_bio2` improved mean score from 1682.6 to 1705.8: +23.3 points, paired 95% bootstrap CI +5.7 to +41.2. Runtime was 16.38 seconds per game. `winner.json` contains the exact winning configuration.

Raw rows are under `final-shards/`; `final-summary/RESULTS.md` is the merged report. Seeds 62001–62200 were used for selection and seeds 63001–65000 for the held-out evaluation.
