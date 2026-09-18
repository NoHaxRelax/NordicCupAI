# Benchmark overview and recorded playback

Run from the frozen benchmark checkout:

```bash
python survival-simulator/scripts/entrapment_benchmark_viewer.py \
  /path/to/runpod-20260918-9059-1000 --port 9061
```

Open http://localhost:9061. The histogram shows final scores for completed games. Click a bar to filter the case table; change bin counts, sort by score, or filter for physical bait and rear occupation. Select a seed for native creature rendering, single-tick stepping, adjustable playback speed, role events, energy inspection and population history. The overview refreshes every 30 seconds while fewer than 1,000 results exist.

Playback reads the actual saved trajectories. It never reruns a seed or substitutes a newly simulated trajectory. Creature bodies and energy bars use the game's `Creature.draw()` function. Agents, predator positions, headings, energy and roles are recorded at every tick. Optional vision shading is reconstructed using the recorded traits and true wall geometry for spectator viewing.

**Recording limits:** fruit, trees, terrain textures, controller observations and actions were not saved in the compact batch traces. They are omitted rather than reconstructed as if recorded. The original 9059 reference replay retains its full-world recording separately.

Raw trajectory files are kept locally under the benchmark directory; their multi-gigabyte size makes them unsuitable for normal Git commits. Summary statistics and per-game results are versioned with the benchmark. The viewer requires the downloaded `raw/pod-*` directories to replay individual cases.
