"""Build final tables and an artifact inventory from completed saved results."""
import json
from pathlib import Path
import statistics as st
from summarize import summarize
from run import ROOT,OUT

paths=[OUT/f'heldout-v5-{phase}.json' for phase in ('generated','fixtures','oracle')]
summary=summarize(paths)
(OUT/'final-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
generated=json.loads(paths[0].read_text())['runs']
fixtures=json.loads(paths[1].read_text())['runs']
oracle=json.loads(paths[2].read_text())['runs']
all_results=[];inventory=[]
for path in sorted(OUT.glob('*.json')):
    value=json.loads(path.read_text())
    if isinstance(value,dict) and 'runs' in value:
        inventory.append(dict(file=path.name,runs=len(value['runs'])));all_results.extend(value['runs'])
(OUT/'experiment-inventory.json').write_text(json.dumps(dict(total_runs=len(all_results),files=inventory),indent=2)+'\n')

lines=['## Corrected held-out results','',
       f'{len(all_results)} local runs are retained across five controller iterations, diagnostics, null comparisons and two held-out protocols. '
       'The tables below use only the final 60 version-5 runs: 24 generated-map runs, 24 orchard runs and 12 separately labelled oracle runs.','',
       '### Default generated maps: eight seeds, 600-second horizon','',
       '| Policy | Mean score | Mean duration | Reached 600 s | Mean harvest energy | Mean captures |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
for mode,label in [('nursery','Existing nursery'),('control','Managed colony, no banishment'),('banish','Managed colony with banishment')]:
    rr=[r for r in generated if r['mode']==mode]
    lines.append(f"| {label} | {st.mean(r['score'] for r in rr):.2f} | {st.mean(r['duration'] for r in rr):.1f} s | {sum(r['duration']>=599.95 for r in rr)}/8 | {st.mean(r['metrics']['harvest_energy'] for r in rr):,.0f} | {st.mean(r['metrics']['captures'] for r in rr):.2f} |")
g=next(r for r in summary if r['scenario']=='generated' and r['treatment']=='banish')
lines+=['',f"Banishment won {g['score_wins']}/8 paired scores and tied {g['score_ties']}/8. Mean score difference: {g['mean_deltas']['score_delta']:+.2f}; descriptive paired-bootstrap 95% interval {g['score_mean_bootstrap95']}. Mean duration difference: {g['mean_deltas']['duration_delta']:+.1f} seconds.",'',
        '| Seed | Control score | Banishment score | Guide selections | Release attempts |',
        '| --- | ---: | ---: | ---: | ---: |']
for seed in sorted({r['seed'] for r in generated}):
    c=next(r for r in generated if r['seed']==seed and r['mode']=='control');b=next(r for r in generated if r['seed']==seed and r['mode']=='banish')
    lines.append(f"| {seed} | {c['score']:.2f} | {b['score']:.2f} | {sum(e['kind']=='guide_selected' for e in b['events'])} | {len(b['releases'])} |")
lines+=['','Seven generated-map pairs matched exactly because the conservative gate never dispatched a guide. All measured benefit came from seed 205: three guide jobs, two controller-reported remote contact losses, and no planned rest-based release. Those contact losses are separate events; this harness does not measure their return delay. This is limited evidence for occasional displacement, not reliable acquisition on ordinary maps. The nursery still had the highest mean score and reached the horizon more often.',
        '', 'A separate **training-only** acquisition ablation removed the conservative gate on seeds 1–3 at 300 seconds. Seed 1 then dispatched five guides, achieved one planned release and gained 2.18 score points, but lost two guides and spent about 777 movement/turning energy. The other two pairs tied with no guide jobs. This suggests acquisition restrictions are a real bottleneck; it does not validate the aggressive variant on held-out maps. [Ablation data](../survival/results/colony_banishment/iteration5-acquisition-ablation.json).']
lines+=['','### Managed orchards: four paired seeds per terrain, 300 seconds','',
        'Differences below are banishment minus the same colony control. Exposure is the percentage-point change in worker time spent within 100 units of any predator.','',
        '| Fixture | Score wins | Mean score difference | Harvest difference | Capture difference | Exposure change |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
for r in summary:
    if 'fixtures' not in r['file']:continue
    d=r['mean_deltas'];lines.append(f"| {r['scenario']} | {r['score_wins']}/4 | {d['score_delta']:+.2f} | {d['harvest_delta']:+,.0f} | {d['capture_delta']:+.2f} | {d['worker_exposure_rate_delta']:+.2f} pp |")
ob=next(r for r in summary if 'fixtures' in r['file'] and r['scenario']=='obstacles')
delta=[p['score_delta'] for p in ob['pairs']]
lines+=['',f"The obstacle result has a median paired gain of {st.median(delta):.2f} points. Its mean is strongly affected by seed 22, where the control died at 162.1 seconds and the banishment colony reached 300. Excluding the largest gain leaves a mean of {st.mean(sorted(delta)[:-1]):.2f} points across the other three cases. One obstacle case had only brief guiding and no formal remote release, so 4/4 is a whole-controller outcome, not proof that every gain came from a completed banishment. This is still a small arranged-fixture sample."]
lines+=['','### Protection, cost and return','',
        '| Evaluation | Guide jobs | Planned remote release attempts | Observed returns / attempts | Median observed return delay | Mean guide time per run | Mean guide movement/turning energy per run | Mean predator-seconds targeting guide |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for rr,label in [(generated,'Generated maps'),(fixtures,'Managed orchards')]:
    rr=[r for r in rr if r['mode']=='banish'];returns=[x['return_t']-x['t'] for r in rr for x in r['releases'] if x['return_t'] is not None]
    jobs=sum(e['kind']=='guide_selected' for r in rr for e in r['events']);releases=sum(len(r['releases']) for r in rr)
    delay=f'{st.median(returns):.1f} s' if returns else 'Not observed'
    lines.append(f"| {label} | {jobs} | {releases} | {len(returns)}/{releases} | {delay} | {st.mean(r['metrics']['guide_seconds'] for r in rr):.1f} s | {st.mean(r['metrics']['guide_move_energy'] for r in rr):.1f} | {st.mean(r['metrics']['guide_target_seconds'] for r in rr):.1f} |")
lines+=['','Unobserved returns are right-censored by extinction or the horizon. A release attempt and surviving guide are not proof of exclusion or permanent occupation. Return radius is 250 around the release-time actual worker centroid, so worker relocation changes its interpretation.','',
        '### Exact-pose diagnostic, kept separate','',
        '| Fixture | Oracle banishment score difference | Score wins |',
        '| --- | ---: | ---: |']
for r in summary:
    if r['treatment']=='oracle':lines.append(f"| {r['scenario']} | {r['mean_deltas']['score_delta']:+.2f} | {r['score_wins']}/2 |")
lines+=['','Exact localization alone did not make the strategy uniformly useful. This diagnostic does not grant geometry, hidden rest or energy to the observation-only policy.','',
        '## Reviewable artifacts','',
        '- [Frozen final protocol](../survival/results/colony_banishment/heldout-v5-freeze.json), [paired summaries](../survival/results/colony_banishment/final-summary.json), and [all-run inventory](../survival/results/colony_banishment/experiment-inventory.json).',
        '- [Generated-map outcomes](../survival/results/colony_banishment/heldout-v5-generated.json), [orchard outcomes](../survival/results/colony_banishment/heldout-v5-fixtures.json), [oracle outcomes](../survival/results/colony_banishment/heldout-v5-oracle.json).',
        '- [Three exact null comparisons](../survival/results/colony_banishment/iteration5-null-verification.json). The score, duration, metrics, events, releases and traces match in all three inactive pairs.',
        '- [Generated-map control replay](../survival/results/colony_banishment/heldout-v5-generated-201-control.json.gz) and [matched banishment-enabled replay](../survival/results/colony_banishment/heldout-v5-generated-201-banish.json.gz). Seed 201 illustrates the policy when no guide is dispatched.',
        '- [Managed forest control replay](../survival/results/colony_banishment/heldout-v5-forest-21-control.json.gz) and [banishment replay](../survival/results/colony_banishment/heldout-v5-forest-21-banish.json.gz).',
        '- [Obstacle orchard banishment replay](../survival/results/colony_banishment/heldout-v5-obstacles-22-banish.json.gz).',
        '- [Active generated seed-205 banishment replay](../survival/results/colony_banishment/final-active-generated-replays-generated-205-banish.json.gz) and [its control](../survival/results/colony_banishment/final-active-generated-replays-generated-205-control.json.gz). These are separate recording reruns; [reproduction check](../survival/results/colony_banishment/active-replay-reproduction.json) compares their game results with the original held-out runs.',
        '- [Source verification](../survival/results/colony_banishment/source-verification.json), [checks](../survival/results/colony_banishment/checks.log), [replay verification](../survival/results/colony_banishment/replay-checks.json).','',
        'Load recordings through the existing Survival Lab file picker. Inspector annotations were wrapped into its existing rule/detail structure after recording; positions, actions, observations, energy and event data were retained.','',
        '## Recommendation and remaining limits','',
        'Keep this as an optional, conservative strategy experiment. The code now performs actual managed harvesting, fueled guide selection, acquisition attempts, repeated displacement, release avoidance and return. The evidence supports temporary displacement as a possible service, but not routine banishment as a general improvement or cheap permanent parking. Preserve the no-banishment colony and existing nursery as controls.','',
        'The main unresolved costs are poor acquisition geometry, guide return routes that can bring pressure back, remote food/aging costs, changing orchard productivity, unknown terrain, and additional predators arriving from an unobserved direction. Only one guide can be active. There is no dedicated return-route guard, no guaranteed opposite-corner navigation, no robust river-isolation planner and no permanent occupation. These were considered; they are not silently claimed as solved.','',
        'A further iteration should pay for a guide only when the predicted extra harvesting/evacuation time exceeds its food and replacement cost, and verify a safe return or independent food patch before departure. It needs new held-out seeds and the full 3000-second horizon. The current bounded samples do not justify replacing the baseline or making a hosted submission.','']
report=ROOT.parent/'docs/survival-colony-banishment.md'
text=report.read_text().split('<!-- HELDOUT_RESULTS -->')[0]
report.write_text(text+'<!-- HELDOUT_RESULTS -->\n\n'+'\n'.join(lines))
print(f'Wrote final tables for {len(all_results)} runs.')
