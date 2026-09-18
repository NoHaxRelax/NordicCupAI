"""Build the review report directly from saved final experiment records."""
from pathlib import Path
import json,statistics,math,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];OUT=ROOT/'results/dedicated_sprinter';DOC=ROOT.parent/'docs/survival-dedicated-sprinter.md'
def rows(name):return json.loads((OUT/(name+'.json')).read_text())['runs']
def table(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in data)+'\n'
def fmt(x):return f'{x:.1f}'
def pct(x):return f'{100*x:.1f}%'
def main():
    controls=rows('controls-final-v5');valid=rows('validation-final-v5');traits=rows('traits-final-v5');protection=rows('protection-v5');replays=rows('replays-final-v5');genes=rows('genetics')
    games={mode:[rows(f'fullgame-{mode}-{seed}'+('-v5' if mode=='caste' else ''))[0] for seed in (1,2,3)] for mode in ('nursery','caste')}
    controller_hash=hashlib.sha256((HERE/'controller.py').read_bytes()).hexdigest()
    ct=[]
    for food in ('none','line','orchard'):
        for r in [x for x in controls if x['scenario']['food']==food]:
            ct.append([food,r['policy'],fmt(r['duration']),pct(r['target_fraction']),fmt(r['longest_occupation_s']),r['food_eaten'],fmt(r['movement_energy'])])
    vt=[]
    for policy in ('adaptive','nofood','predictive'):
        rr=[x for x in valid if x['policy']==policy]
        vt.append([policy,fmt(statistics.median(x['duration'] for x in rr)),pct(statistics.mean(x['target_fraction'] for x in rr)),fmt(statistics.median(x['longest_occupation_s'] for x in rr)),sum(x['longest_occupation_s']>=60 for x in rr),fmt(statistics.median(x['food_eaten'] for x in rr))])
    pt=[]
    for mode in ('none','adaptive','predictive'):
        rr=[x for x in protection if x['mode']==mode]
        pt.append([mode,sum(x['workers_alive'] for x in rr),sum(x['worker_captures'] for x in rr),fmt(sum(x['worker_seconds'] for x in rr))])
    tt=[]
    for r in traits:
        c=r['scenario'];tt.append([c['walk'],c['capacity'],c['energy'],c['food'],fmt(r['duration']),pct(r['target_fraction']),r['food_eaten']])
    gt=[]
    for i,seed in enumerate((1,2,3)):
        b=games['nursery'][i];c=games['caste'][i]
        gt.append([seed,fmt(b['score']),fmt(c['score']),fmt(c['score']-b['score']),fmt(b['duration']),fmt(c['duration']),c['births'],len({p['agent_id'] for p in c['stats']['preparation']})])
    founder=replays[0]; rr=[x for x in valid if x['policy']=='predictive']
    tp=sum(x['rest_inference']['true_positive'] for x in rr);fp=sum(x['rest_inference']['false_positive'] for x in rr);fn=sum(x['rest_inference']['false_negative'] for x in rr)
    summary=dict(primary_runs=85,controller_sha256=controller_hash,controls=ct,heldout=vt,protection=pt,fullgame=gt,mean_scores={m:statistics.mean(r['score'] for r in rs) for m,rs in games.items()},rest_inference_precision=tp/max(1,tp+fp),rest_inference_recall=tp/max(1,tp+fn),heldout_max_pose_error=max(r['max_pose_error'] for r in rr))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# Dedicated single-predator bait: executable next iteration

17 September 2026. Personal Nordic AI Cup research. **The useful new strategy is a cheap hold inside the predator's turning circle, with controlled food excursions. Ordinary founders can perform it; forced speed breeding is not a prerequisite.** The controller and local experiments are complete. This is an opt-in research policy, not a demonstrated solution to the full game.

The representative final founder starts with **150 energy**, holds the selected predator continuously for **{founder['duration']:.1f} seconds**, consumes **{founder['food_eaten']} fruits**, and spends **{founder['movement_energy']:.2f} movement energy**. It eventually starves. The old adaptive controller lasts 9.5 seconds in the same orchard. With no food the new controller holds for 68.7 seconds, versus 2.2 seconds for the old controller. These comparisons use original physics, cached observations, physical boundaries and finite resources.

## Scope and evidence

All work stayed in the assigned research/results directories and this report. Original source is `survival/vendor/survival-simulator`, commit `acfc31a4003a5f91bf11032a02cd98c178ddbd7e`; interpreter is `survival/.venv/bin/python`. No competition API, hosted validation, submission, publishing, pushes, vendor changes, worker-policy edits, debugger changes or new renderer. Water baiting, wall acquisition and two-bait handoff are outside this experiment.

The final evidence comprises **85 primary engine runs**: 9 matched controls, 36 held-out cases, 16 trait/energy cases, 18 fixed-horizon worker-protection runs and 6 generated-map games. Five thousand optimistic mutation searches and seven final representative replays supplement them. Earlier controller screens and failed variants are preserved separately.

**Geometry construction is privileged; action decisions are observation-only.** Controlled arenas are 1600×1200 forest with 30-unit boundary walls. The default bait is at (800,600), predator 60 units west, facing east with 102 energy; bait faces west. Agent senescence threshold is 90 seconds unless explicitly varied. Initial predator wake state is arranged, not inferred from gameplay. No new predators or trees are added in these fixtures. Original fruit growth, aging, tree death, collision, hearing, predator control and energy rules run unchanged.

Food fixtures: `line` contains four mature fruits east of the bait at offsets 140/300/460/620. `orchard` contains eight ordinary trees aged 20 seconds on a radius-100 ring plus six mature fruits on a radius-65 ring. The trees produce fruit using the engine's stochastic rules and then die. These are generous **finite arranged orchards**, not free ongoing nutrition or evidence that a randomly spawned bait can acquire such a patch. Final generated-map tests use the unmodified default map generator and all normal spawns.

## What the controller does

[`controller.py`](../survival/research/dedicated_sprinter/controller.py) takes one ordinary agent DTO and simulation time. It never receives true pose, predator identity, hidden predator energy/rest, fruit age, tree age, a true terrain map or assigned world coordinates.

1. Reconstructs a local frame from self motion, current biome and remembered static edges/trees. Fruit positions are remembered for interception but excluded from pose correction because fruit has no identity and can disappear. Facing and movement directions remain independent.
2. Infers predator motion from consecutive cached sightings and rest after two nearly stationary sightings. Advances the delayed predator pose before evaluating the next action. It also considers a full-speed waking predator instead of treating inferred rest as a guarantee.
3. Scores finite single-action candidates for energy, safety, hearing contact, observed obstacles, known slow terrain and progress toward remembered fruit. Requested movement is clamped to the observed sprint trait and current cutoff. No repeated-action trick is used.
4. In a cheap hold, targets the interior of the predator's turning circle. A predator advancing 15 while turning 0.3 radians follows a discrete circle of radius `15 / (2 sin(0.15)) ≈ 50.2`. A bait that safely gets inside that circle can stay selected within hearing range while barely moving. The controller recentres as observed speed changes.
5. Switches to food-seeking curved pursuit when fruit is remembered and fuel is below `max(150, 0.7 × capacity)`. It maintains a cost for leaving 58-unit predicted hearing range and a 24-unit predicted capture margin. These are soft objectives, not a proof of safety. Resting windows create opportunities to collect food.

Energy balance and nutrition caps were separately audited for 67 final controlled/replay records, and corrected terminal aging accounting was checked against a fresh run. The prediction function matched 200 original clear-ground predator actions, across headings, ranges and both normal speed and the low-energy cutoff, to floating-point error below 1.2e-13. A final correction preserved heading while modelling rest and used the source's requested-speed pivot geometry even at low actual speed. Safety still depends on imperfect observation association and terrain/collision inference.

On final held-out cases, inferred-rest precision was {pct(tp/max(1,tp+fp))} and recall {pct(tp/max(1,tp+fn))}. The rest classifier can confuse a blocked predator with sleep. No hidden state corrects policy decisions; hidden fields are used only for diagnostic metrics.

## Matched training controls

All rows below start with 150 energy, walking 10, sprinting 20 and capacity 500. `adaptive` is the prior direct escape policy. `nofood` uses the new controller but disables deliberate food pursuit; incidental eating is still possible. `predictive` is the final automatic hold/forage controller.

{table(['Food','Policy','Lifetime s','Active target share','Longest occupied s','Fruits eaten','Movement energy'],ct)}

“Active target share” counts real predator selections before its original movement, including ticks when it wakes. “Longest occupied” is the longest uninterrupted interval without an active selection away from the bait, including intervening rest time. Death ends the interval. Survival without selection does not count as occupation. No-food and line results expose why lifetime alone is inadequate: food detours can lose attention even while the bait survives.

The representative orchard founder receives {founder['food_received']:.1f} nutrition after capacity clipping, from {founder['nutrition']:.1f} available nutrition in eaten fruits. Movement costs {founder['movement_energy']:.1f}, turning {founder['turn_energy']:.1f}, and passive/aging drain {founder['passive_energy']:.1f}. Unconsumed fruit and nutrition wasted at capacity are not credited as fuel.

## Untuned geometry, heading, fuel and terrain

The final split was drawn with fixed generator seed 403971 and engine seeds 401–412. It varies position, rotation, heading ±2 radians, separation 50–85, predator energy 40–200, bait energy 150/250/400, senescence 60–120, obstacles, desert and forest/swamp boundaries. Parameters are stored per run. They were not used for parameter selection. Each of the three controllers receives the same construction parameters; controller-dependent fruit interactions can alter subsequent shared RNG consumption.

{table(['Policy','Median lifetime s','Mean active target share','Median longest occupied s','Runs with ≥60 s occupation /12','Median fruits'],vt)}

These tests validate a held encounter and some local acquisition, not a guaranteed search for predators or orchards. One split-terrain case actually entered swamp; the other two stayed on the forest side. Seed 407 was captured at 64.5 seconds with 450.6 energy still available, so fuel reserve alone does not make the controller safe. The largest measured odometry error was {summary['heldout_max_pose_error']:.1f} units. Only the current and previously visited biomes are known; an unseen terrain transition remains dangerous. Observation payloads lack predator IDs, so nearest-neighbour motion inference can switch individuals when several predators approach.

## Worker protection at a common horizon

Six new geometries/headings, each with two stationary 150-energy workers, were run for the same **90 seconds**, including after a bait died. Worker starvation and capture are separated; no worker starved in this set. This avoids rewarding a failed bait merely because its experiment stopped early.

{table(['Policy','Workers alive /12','Worker captures /12','Worker-seconds /1080'],pt)}

This is a controlled protection result, not a forage-colony score. Workers are arranged away from the initial chase. A worker entering the predator's path can become the closest target; the bait has no privileged control over target selection.

## Small specialist caste and preparation cost

[`colony.py`](../survival/research/dedicated_sprinter/colony.py) keeps the existing nursery worker policy and limits bait roles to at most two and at most one-third of the living colony. It attempts one bait per predator observation cluster using relative Agent links to align local frames. Disconnected observation components can still duplicate a predator. Roles expire on death or lost contact; there is no coordinated handoff. Readiness requires observed sprint speed ≥18 and fuel at least 70 above cutoff. Selecting the best nearby prepared agent is supported; a dedicated breeding programme is not justified by these runs.

{table(['Walk','Capacity','Initial energy','Food','Lifetime s','Active target share','Fruits'],tt)}

A normal newborn begins at 75 while capacity 500 implies a 100-energy sprint cutoff. The final newborn replay dies after 0.4 seconds despite nearby orchard resources. Lower capacity 300 gives a 60 cutoff and only 15 energy of initial sprint reserve. It does not guarantee safe deployment. Cheap walking speed can help, but selection for life alone is unsafe: the replay using an actual sampled mutant (walk 13.934, sprint 20, capacity 309.452) lives 112.1 seconds yet keeps attention for only 43.0% of active ticks.

The original `Environment.spawn_agent` was used for 1000 searches per target. The optimistic nursery has one best parent, unlimited replacement food, no elapsed game time, no senescence and no predators. Failed children are removed. Therefore these are **optimistic mutation-effort estimates**, not feasible preparation demonstrations.

{table(['Mutation target','Births p10','Births median','Births p90','Median parent birth debits'],[[r['target'],r['p10'],fmt(r['median']),r['p90'],fmt(r['median_parent_birth_energy'])] for r in genes])}

Each birth debits its parent 100 energy and endows the child with 75. The debit column is cumulative parent spending, not net energy destroyed or an exact external food requirement. Useful failed offspring, replacement-parent energy, collection costs, time and mortality need a real nursery simulation. The search conserves neither time nor food, and cannot establish that 44 births can be produced before an encounter.

A founder starts at 150. Preparing a 500-energy founder needs at least 350 added nutrition, at least six fully mature 60-energy fruits before movement/living costs. Raising a newborn from 75 to 150 needs 75 added nutrition, at least two mature fruits. Filling larger capacity is useful only if that food is actually available. The final controller's headline arranged result uses the ordinary founder and needs no mutation preparation.

The 114.7-second window is not permanent occupation. Even if it could be repeated with perfect handoffs, a single predator over 3000 seconds would require at least 27 bait lifetimes, hence 26 replacement births and 2600 parent birth debits, plus food and safe staging. No such replacement chain was executed. Multiple predators and aging increase the practical burden.

## Generated-map games

Three paired default generated maps ran to extinction or the normal 3000-second horizon. The baseline is the unchanged nursery policy; the candidate adds the bounded specialist assignment layer and final single-bait controller. Births and fuel are obtained through ordinary gameplay; no prepared agents, trait injection, arranged food or hidden geometry are supplied. There is no forced specialist breeding in these games.

{table(['Seed','Nursery score','Caste score','Difference','Nursery survival s','Caste survival s','Caste births','Distinct assigned baits'],gt)}

Mean score: nursery {summary['mean_scores']['nursery']:.1f}, caste {summary['mean_scores']['caste']:.1f}. The candidate wins score on **one of three seeds**, and none of the six games completes 3000 seconds. This sample does not establish a reliable score improvement. Assignment episodes include reacquisition of the same agent, so they must not be read as replacement births. Logs preserve assignment time, energy, age and traits. An earlier unbounded assignment prototype reached four baits and performed badly on seed 1; it is preserved in `fullgame-caste-1-v3.json`, excluded from the final comparison.

The original engine shares its RNG between ecology, movement and reproduction, so policy changes diverge future maps/events. These are paired starting seeds, not matched future random events. Local macOS execution also does not prove equivalence with hosted Linux scoring.

## Failed hypotheses and remaining limits

- Straight adaptive sprinting wastes fuel and cannot deliberately harvest a patch. It remains a useful paired control.
- Curved pursuit alone improved occupation but spent hundreds of movement energy. The turning-circle hold reduced that cost substantially.
- Weak food attraction plus quiet facing often increased lifetime by releasing the predator. Strong food attraction could similarly destroy retention. Those screens remain in `center-food-screen.json`; the final controller prioritizes contact and retains normal facing.
- Fruit-based pose correction produced spurious shifts when unlabelled food changed. Final localization uses static trees/edges only. Unknown collisions and ambiguous landmarks remain possible.
- Moving less is insufficient if aging dominates. Finite trees die and fruit rots. The representative founder eventually starves despite successful holding and collection.
- Fast or low-capacity mutants are not automatically better baits. The actual-mutant replay demonstrates high survival with poor occupation. Prefer readiness and measured retention over a single trait threshold.
- The assignment layer cannot yet guarantee one globally unique predator per bait under disconnected observations. Contact churn, distractors, several simultaneous predators and replacement staging remain unsolved.

The concrete deliverable is the tested observation-only controller, an optional small-caste wrapper, reproducible scenarios, diagnostics and replays. The next justified deployment gate is reliable assignment/acquisition and replacement under generated-map observations; the holding controller itself has substantially stronger evidence than the earlier 39.1-second, 500-energy/four-fruit demonstration.

## Artifacts and reproduction

Final controller SHA-256: `{controller_hash}`.

Primary final records are `controls-final-v5.json`, `validation-final-v5.json`, `traits-final-v5.json`, `protection-v5.json`, `fullgame-caste-{{1,2,3}}-v5.json`, the three `fullgame-nursery-*.json` controls, `genetics.json`, `replays-final-v5.json`, `summary.json` and `verification.json`, all under `survival/results/dedicated_sprinter/`. Earlier v1–v4 screens are development evidence and are not pooled into final statistics. Source snapshots are retained as `controller_v1.py` through `controller_v4.py`.

From the repository root:

```sh
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/controls.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/validation.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/protection.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/fullgame.py --mode caste --seed 1
survival/.venv/bin/python survival/research/dedicated_sprinter/fullgame.py --mode nursery --seed 1
survival/.venv/bin/python survival/research/dedicated_sprinter/genetics.py
DEDICATED_RUN_TAG=v5 survival/.venv/bin/python survival/research/dedicated_sprinter/replays.py
survival/.venv/bin/python survival/research/dedicated_sprinter/verify.py
survival/.venv/bin/python survival/research/dedicated_sprinter/report.py
```

Repeat the two full-game commands for seeds 2 and 3. All commands run locally. `DEDICATED_CONTROLLER=controller_v2` selects an archived controller for development comparisons.

Load these files with the existing Survival Lab viewer's file loader:

- [`final-founder-orchard.json.gz`](../survival/results/dedicated_sprinter/final-founder-orchard.json.gz): 150-energy founder, actual food excursions and eventual starvation.
- [`final-founder-no-food.json.gz`](../survival/results/dedicated_sprinter/final-founder-no-food.json.gz): cheap holding without nutrition.
- [`final-protected-workers.json.gz`](../survival/results/dedicated_sprinter/final-protected-workers.json.gz): fixed-horizon protection of two workers.
- [`final-specialist-orchard.json.gz`](../survival/results/dedicated_sprinter/final-specialist-orchard.json.gz): synthetic walk-12/capacity-300 specialist and its failure.
- [`final-newborn-cutoff.json.gz`](../survival/results/dedicated_sprinter/final-newborn-cutoff.json.gz): normal newborn fails before feeding.
- [`final-real-mutant.json.gz`](../survival/results/dedicated_sprinter/final-real-mutant.json.gz): feasible sampled traits, survival exceeding occupation.

- [`final-heldout-capture.json.gz`](../survival/results/dedicated_sprinter/final-heldout-capture.json.gz): untuned obstacle/heading case that loses a well-fuelled bait.

Replays use the existing `survival/debugger/recorder.py` format, record cached action observations and diagnostic truth separately, and do not alter the viewer or manifest.
'''
    audit_path=OUT/'replay-backfill-audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text());counts=audit.get('counts',{})
        text+="\n## Replay coverage and immutable new runs\n\n"
        text+=f"The replay audit covers {len(audit['cases'])} saved completed game-run records, including the earlier log-only validation sweep. Coverage: {counts}. Existing recordings and historical metrics are preserved. Newly reproduced runs are labelled **New reproduction**, with their original result file, row, controller version and original outcome in metadata. They are fresh local executions, never recovered original footage. Some outcomes differ from the historical tables; each reproduction keeps its own measured result in its receipt. The copied food-v1 metrics file actually matches the v2 results and source hash, so the audit corrects that version association without rewriting the old file.\n\n"
        text+="Every new arranged, protection and generated-map run now records by default. Initial state and every timestep are passed to the existing ReplayRecorder; sampling stores ordinary frames every 0.5 seconds (1 second for full games) while critical events retain exact frames. Key demonstrations use original native rendering. Replays go to `survival/results/dedicated_sprinter/replays/`, per-run receipts to `runs/`, and unique aggregate outputs to `summaries/`. No run overwrites an earlier recording or aggregate.\n\n"
        text+="Five aggregate mutation-distribution records are excluded from gameplay footage: those searches have no advancing game clock or environment timesteps. The early v3 full-game assignment wrapper was not archived; its reproduction uses an explicitly labelled reconstruction plus the archived v3 controller. Unlogged diagnostic reruns cannot be individually recovered. Other console logs mirror the saved result rows and are not counted twice.\n\n"
        text+="Audit: [`replay-backfill-audit.json`](../survival/results/dedicated_sprinter/replay-backfill-audit.json). Reproduction commands: `backfill.py --inventory`, `backfill.py --worker 0 --workers 1`, and `backfill.py --refresh`, using the project interpreter. Existing per-run receipts make workers resumable. Read-only live-catalog verification: `check_catalog.py --require-complete`. Survival Lab discovers complete recordings automatically; its offline HTML export remains a separate snapshot.\n"
    DOC.write_text(text);print(DOC)
if __name__=='__main__':main()
