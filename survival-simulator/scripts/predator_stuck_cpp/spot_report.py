"""Test explicit trapping hypotheses and validate geometry rules on held-out maps."""
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

NUMERIC = ['seed', 'predator_id', 'is_case', 'spawned_overlapping', 'active_ticks',
    'blocked_fraction', 'fallback_fraction', 'edge_fraction', 'wander_fraction',
    'quarter_turn_fraction', 'stationary_fraction', 'selected_edge_close_fraction',
    'pose_period', 'position_period', 'best_pose_return_fraction', 'four_step_position_fraction',
    'escaped_after_detection', 'candidate_reconstruction_mismatch', 'late_logged_all_blocked',
    'bbox_width', 'bbox_height', 'biome_count', 'late_x', 'late_y', 'late_heading', 'late_step',
    'late_nearest_clearance', 'late_second_clearance', 'late_third_clearance',
    'late_nearby_obstacles_2', 'late_nearby_obstacles_15', 'late_overlap_count',
    'late_nearest_boundary_clearance', 'late_nearest_internal_clearance',
    'late_expanded_corner_distance', 'late_perpendicular_faces_2', 'late_perpendicular_faces_15',
    'late_opposing_faces_15', 'late_boundary_corner_15', 'late_free_step_directions_36',
    'late_borderline_step_directions_36', 'late_sampled_free_direction_fraction',
    'late_step_circle_inside_one_obstacle', 'entry_nearest_clearance', 'entry_second_clearance']


def load_windows(path):
    chunks = []; buffer = []
    texts = {k: [] for k in ('classification', 'biome', 'shard', 'trace', 'spot_id')}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            buffer.append([float(row[k]) for k in NUMERIC])
            for key in texts:
                texts[key].append(row[key])
            if len(buffer) == 4096:
                chunks.append(np.asarray(buffer)); buffer = []
    if buffer:
        chunks.append(np.asarray(buffer))
    array = np.concatenate(chunks)
    return {k: array[:, i] for i, k in enumerate(NUMERIC)} | {
        k: np.asarray(v) for k, v in texts.items()}


def hypotheses(d):
    """Predeclared thresholds; no tuning on held-out maps."""
    result = {}
    def add(key, label, mask, geometry=True):
        result[key] = dict(label=label, mask=mask, geometry_only=geometry)
    add('overlap', 'Overlaps an expanded collision rectangle', d['late_overlap_count']>0)
    add('clearance_2', 'Nearest collision clearance is at most 2 units', d['late_nearest_clearance']<=2)
    add('clearance_step', 'Nearest collision clearance is at most one step', d['late_nearest_clearance']<=d['late_step'])
    add('clearance_15', 'Nearest collision clearance is at most 15 units', d['late_nearest_clearance']<=15)
    add('two_near_2', 'At least two collision rectangles are within 2 units', d['late_second_clearance']<=2)
    add('two_near_step', 'At least two collision rectangles are within one step', d['late_second_clearance']<=d['late_step'])
    add('two_near_15', 'At least two collision rectangles are within 15 units', d['late_second_clearance']<=15)
    add('three_near_15', 'At least three collision rectangles are within 15 units', d['late_third_clearance']<=15)
    add('boundary_near', 'Within 15 units of an expanded world boundary', d['late_nearest_boundary_clearance']<=15)
    add('internal_near', 'Within 15 units of an expanded internal obstacle', d['late_nearest_internal_clearance']<=15)
    add('world_corner', 'Within 15 units of two expanded world boundaries', d['late_boundary_corner_15']>0)
    add('expanded_corner', 'Within 15 units of an expanded rectangle corner', d['late_expanded_corner_distance']<=15)
    add('perpendicular_faces_2', 'Perpendicular collision faces within 2 units', d['late_perpendicular_faces_2']>0)
    add('perpendicular_faces_15', 'Perpendicular collision faces within 15 units', d['late_perpendicular_faces_15']>0)
    add('opposing_faces', 'Opposing collision faces each within 15 units', d['late_opposing_faces_15']>0)
    add('slow_step', 'Biome-scaled step is at most 15/sqrt(2) units', d['late_step']<=15/math.sqrt(2))
    add('no_sampled_free_step', 'No legal endpoint in the current 36-direction step grid', d['late_free_step_directions_36']==0)
    add('some_sampled_free_step', 'At least one legal endpoint in the current step grid', d['late_free_step_directions_36']>0)
    add('few_free_steps', 'At most half the 36 step directions have legal endpoints', d['late_free_step_directions_36']<=18)
    add('deep_overlap', 'Whole step circle lies strictly inside one collision rectangle', d['late_step_circle_inside_one_obstacle']>0)
    diagonal_error = np.abs(np.remainder(d['late_heading'], math.pi/2)-math.pi/4)
    add('diagonal_heading', 'Heading is within 1 degree of a map diagonal', diagonal_error<=math.pi/180)
    add('spawn_overlap', 'Initial spawn overlaps an obstacle', d['spawned_overlapping']>0, False)
    add('any_avoidance', 'At least one active tick uses edge avoidance', d['edge_fraction']>0, False)
    add('only_avoidance', 'Every active tick uses edge avoidance', d['edge_fraction']==1, False)
    add('no_wandering', 'No active tick uses random wandering', d['wander_fraction']==0, False)
    add('quarter_turns', 'At least 95% of active ticks turn by 90 degrees', d['quarter_turn_fraction']>=.95, False)
    add('any_fallback', 'At least one active tick uses a successful fallback', d['fallback_fraction']>0, False)
    add('moving_pose_cycle', 'Moving path has a 1–16 step pose recurrence in its final 128 active ticks',
        (d['pose_period']>0)&(d['stationary_fraction']<.05), False)
    add('position_cycle', 'Final path has a 1–16 step position recurrence', d['position_period']>0, False)
    add('one_biome', 'All active ticks occur in one biome', d['biome_count']==1, False)
    return result


def rate(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def example(d, i):
    return dict(seed=int(d['seed'][i]), predator_id=int(d['predator_id'][i]),
        shard=str(d['shard'][i]), trace=str(d['trace'][i]), spot_id=str(d['spot_id'][i]),
        x=float(d['late_x'][i]), y=float(d['late_y'][i]), heading_radians=float(d['late_heading'][i]),
        nearest_clearance=float(d['late_nearest_clearance'][i]),
        second_clearance=float(d['late_second_clearance'][i]),
        pose_period=int(d['pose_period'][i]), step=float(d['late_step'][i]))


def evaluate_rule(mask, cases, controls, selection, weights):
    positive = cases & selection; negative = controls & selection
    hits = int((mask & positive).sum()); false_positives = int((mask & negative).sum())
    total = int(positive.sum()); control_total = int(negative.sum())
    weighted = rate(weights[positive & mask].sum(), weights[positive].sum())
    specificity = rate(control_total-false_positives, control_total)
    return dict(cases=total, cases_matching=hits, case_coverage=rate(hits, total),
        spot_weighted_case_coverage=weighted, controls=control_total,
        controls_matching=false_positives, control_specificity=specificity,
        balanced_score=(weighted+specificity)/2 if weighted is not None and specificity is not None else None)


def rank_rules(hyp, d, cases, controls, weights):
    train = d['seed'].astype(np.int64) % 5 != 0
    test = ~train
    keys = [k for k, h in hyp.items() if h['geometry_only']]
    candidates = [(k, (k,), 'single') for k in keys]
    for a, b in itertools.combinations(keys, 2):
        for operation in ('AND', 'OR'):
            candidates.append((f'{a} {operation} {b}', (a, b), operation))
    ranked = []
    seen = set()
    for key, terms, operation in candidates:
        mask = hyp[terms[0]]['mask']
        if operation == 'AND': mask = mask & hyp[terms[1]]['mask']
        if operation == 'OR': mask = mask | hyp[terms[1]]['mask']
        stats = evaluate_rule(mask, cases, controls, train, weights)
        if stats['balanced_score'] is None:
            continue
        if stats['case_coverage'] < .5 or stats['control_specificity'] < .5:
            continue
        signature = np.packbits(mask[train]).tobytes()
        if signature in seen: continue
        seen.add(signature)
        ranked.append((stats['balanced_score'], key, terms, operation, stats))
    ranked.sort(key=lambda x: (-x[0], len(x[2]), x[1]))
    winners = []
    for _, key, terms, operation, training in ranked[:5]:
        mask = hyp[terms[0]]['mask']
        if operation == 'AND': mask = mask & hyp[terms[1]]['mask']
        if operation == 'OR': mask = mask | hyp[terms[1]]['mask']
        winners.append(dict(id=key, terms=terms, operation=operation,
            label=(' '+operation+' ').join(hyp[t]['label'] for t in terms),
            training=training, held_out=evaluate_rule(mask, cases, controls, test, weights)))
    return winners


def percent(value):
    return 'n/a' if value is None else f'{100*value:.2f}%'


def make_report(output):
    coverage = json.loads((output/'coverage.json').read_text())
    settings = json.loads((output/'settings.json').read_text())
    d = load_windows(output/'windows.csv')
    cases = d['is_case'] == 1; controls = ~cases
    hyp = hypotheses(d)
    cohorts = dict(all_findings=cases, moving_confinement=d['classification']=='moving_confinement',
        all_active_moves_blocked=d['classification']=='all_active_moves_blocked',
        mixed_blocking=d['classification']=='mixed_blocking')
    spot_counts = Counter(d['spot_id'][cases])
    weights = np.asarray([1/spot_counts[s] if s else 0. for s in d['spot_id']])
    condition_rows = []; counterexamples = []
    for cohort, members in cohorts.items():
        for key, h in hyp.items():
            matching = int((members & h['mask']).sum())
            exceptions = members & ~h['mask']
            condition_rows.append(dict(cohort=cohort, hypothesis=key, description=h['label'],
                geometry_only=h['geometry_only'], cases=int(members.sum()), matching=matching,
                exceptions=int(exceptions.sum()), coverage=rate(matching, int(members.sum())),
                universal_in_this_sample=bool(members.any() and not exceptions.any()),
                controls_matching=int((controls & h['mask']).sum()), controls=int(controls.sum()),
                spot_weighted_coverage=rate(weights[members & h['mask']].sum(), weights[members].sum())))
            for i in np.flatnonzero(exceptions)[:5]:
                counterexamples.append(dict(cohort=cohort, hypothesis=key, **example(d, i)))
    with (output/'conditions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(condition_rows[0])); writer.writeheader(); writer.writerows(condition_rows)
    with (output/'counterexamples.csv').open('w', newline='', encoding='utf-8') as f:
        if counterexamples:
            writer = csv.DictWriter(f, fieldnames=list(counterexamples[0])); writer.writeheader(); writer.writerows(counterexamples)
    rules = rank_rules(hyp, d, cases, controls, weights)
    distributions = {}
    for name, members in cohorts.items():
        distributions[name] = dict(count=int(members.sum()),
            pose_period_counts={str(int(k)):int(v) for k,v in Counter(d['pose_period'][members]).items()},
            biome_counts=dict(Counter(str(v) for v in d['biome'][members])),
            features={key: dict(zip(('minimum', 'p05', 'median', 'p95', 'maximum'),
                map(float, np.quantile(d[key][members], [0, .05, .5, .95, 1]))))
                for key in ('late_nearest_clearance','late_second_clearance','late_step',
                            'quarter_turn_fraction','edge_fraction','fallback_fraction',
                            'late_sampled_free_direction_fraction') if members.any()})
    moving = cohorts['moving_confinement']; blocked = cohorts['all_active_moves_blocked']
    findings = dict(moving_with_free_step=int((moving & hyp['some_sampled_free_step']['mask']).sum()),
        moving_with_pose_cycle=int((moving & hyp['moving_pose_cycle']['mask']).sum()),
        fully_blocked_deep_inside_one_obstacle=int((blocked & hyp['deep_overlap']['mask']).sum()),
        fully_blocked_with_sampled_free_direction=int((blocked & (d['late_sampled_free_direction_fraction']>0)).sum()),
        moving_without_perpendicular_faces_15=int((moving & ~hyp['perpendicular_faces_15']['mask']).sum()),
        moving_without_two_obstacles_15=int((moving & ~hyp['two_near_15']['mask']).sum()),
        moving_without_internal_obstacle_15=int((moving & ~hyp['internal_near']['mask']).sum()),
        moving_with_free_step_but_later_no_escape=int((moving & hyp['some_sampled_free_step']['mask'] & (d['escaped_after_detection']==0)).sum()),
        borderline_candidate_windows=int((d['late_borderline_step_directions_36']>0).sum()),
        candidate_reconstruction_mismatches=int(d['candidate_reconstruction_mismatch'].sum()))
    summary = dict(coverage=coverage, findings=findings, conditions=condition_rows,
        geometry_rules=rules, distributions=distributions,
        observed_universal_conditions={name:[r['hypothesis'] for r in condition_rows
            if r['cohort']==name and r['universal_in_this_sample']] for name in cohorts},
        methodology=dict(pose='Geometry measured immediately before the last active move in the detected interval; entry features also retained.',
            period='Smallest lag 1..16 with >=99% position/heading returns in the final 128 active ticks; 1e-6 position and radian tolerances.',
            spot='Per-map connected components of late path centers separated by <= cluster_radius; a component can chain beyond that distance.',
            controls='One lowest-index unflagged predator per map, final 60 seconds. Not random or matched; rates are descriptive, not population risks.',
            training='Geometry-only single predicates and two-predicate AND/OR rules. Choose up to five using seeds % 5 != 0; evaluate on seeds % 5 == 0.',
            weighting='Each approximate spot has unit total case weight; report raw predator coverage too. No precision/prevalence estimate from this case-control sample.',
            limits='Finite observed invariants are not universal necessities. Geometry at an occupied trap cannot by itself establish reachability or capture probability. Recurrence tolerances and 600-second censoring prevent a permanence proof.'),
        feature_settings=settings, report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    def condition(cohort, key):
        return next(r for r in condition_rows if r['cohort']==cohort and r['hypothesis']==key)
    lines = ['# Conditions at recorded predator trap locations', '',
        f"Analyzed **{coverage['cases']:,} findings from {coverage['games']:,} maps**, "
        f"plus {coverage['controls']:,} unflagged control windows. Grouping nearby late-path centers "
        f"produced **{coverage['spots']:,} approximate spots** (link distance {settings['cluster_radius']:g} units).", '',
        '## What makes trapping possible', '',
        '**The common geometry in moving traps is nearby multiple obstacles.** '
        f"Of {int(moving.sum()):,} moving-confinement cases, "
        f"{findings['moving_without_two_obstacles_15']:,} lacked two collision rectangles within 15 units, and "
        f"{findings['moving_without_internal_obstacle_15']:,} lacked an internal obstacle within that distance. "
        'A world boundary can be one of the rectangles. Distances are signed clearances after '
        'expanding obstacles by predator radius, measured just before the final active move. '
        'This is an observed requirement of the recorded moving cases, not proof that two '
        'nearby obstacles are sufficient to trap any approaching predator.', '',
        f"**Moving traps usually have legal movement steps available.** At the late measurement pose, "
        f"{findings['moving_with_free_step']:,} of {int(moving.sum()):,} moving-confinement cases "
        'had at least one legal endpoint among the native 36 distinct candidate directions. '
        'The controller can keep choosing a closed path despite those alternatives. '
        f"{findings['moving_with_pose_cycle']:,} moving cases showed a repeating position-and-heading "
        'pattern in their final active moves.', '',
        '**One sufficient condition for complete blocking is deeper overlap than the step length.** '
        'Let L be the biome-scaled step length and expand each obstacle by predator radius. '
        'If the predator center is inside one expanded rectangle and its distance to every face '
        'is greater than L, every possible step endpoint remains inside that rectangle. '
        'No heading or fallback angle can move it out. Under this no-agent, static-map walking '
        'behavior, staying at the same position preserves that obstruction. '
        f"This condition held for {findings['fully_blocked_deep_inside_one_obstacle']:,} of "
        f"{int(blocked.sum()):,} fully blocked findings at the measured late pose.", '',
        'The native spawn check also explains how an overlapping predator can be created: '
        '`spawn_predator` passes `(x, y, size, size)` to a rectangle test that treats `(x, y)` '
        'as the upper-left corner, while movement treats the same point as the center and '
        'expands obstacles by `size` in every direction. The spawn test can therefore accept '
        'a position rejected by the movement collision geometry. Smaller biome-scaled steps '
        'make deep overlap harder to leave. This code-level mismatch is separate from the '
        'moving-cycle mechanism, which also occurs after legal spawns.', '',
        'More generally, an active move is blocked exactly when every endpoint the native '
        'candidate search tries is rejected by its local collision test. The analyzer independently '
        'reconstructs the 36 distinct endpoints against nearby map rectangles and records '
        'borderline floating-point contacts separately. '
        f"There were {findings['candidate_reconstruction_mismatches']:,} non-borderline mismatches "
        'with the logged blocked/successful outcome '
        f"({findings['borderline_candidate_windows']:,} borderline windows were excluded from that check). "
        f"In {findings['fully_blocked_with_sampled_free_direction']:,} fully blocked cases, a finer "
        'one-degree heading sample found a legal endpoint even though all directions in the '
        'native 10-degree grid were blocked at that pose. A legal endpoint alone does not '
        'prove there is a path out of the 15-unit confinement circle.', '',
        '**A repeating moving trap additionally depends on the controller state.** The edge '
        'avoidance branch is deterministic; when the selected visible-edge distance minus '
        'predator radius is at most two units, its requested turn reaches 90 degrees. '
        'Accepted fallback movement can differ from the heading change. '
        'If the active-move transition returns to the same position, heading and decision-relevant '
        'state with the same static surroundings, it can repeat indefinitely. Measured approximate '
        'recurrence supports this mechanism but does not prove exact recurrence forever.', '',
        '## Candidate requirements tested against every finding', '',
        'A counterexample disproves necessity in the recorded data. Zero counterexamples means '
        'only that a condition held throughout this sample. Geometry below uses the late pose, '
        'not a claim about every point of the whole interval.', '',
        '| Candidate requirement | All findings matching | Moving findings matching | Fully blocked matching | Unflagged controls matching |',
        '| --- | ---: | ---: | ---: | ---: |']
    display = ['world_corner','perpendicular_faces_15','opposing_faces','two_near_15',
               'clearance_2','overlap','deep_overlap','diagonal_heading','quarter_turns',
               'only_avoidance','any_fallback','moving_pose_cycle']
    for key in display:
        row = condition('all_findings', key)
        lines.append('| '+hyp[key]['label']+' | '+' | '.join(percent(condition(name,key)['coverage'])
            for name in ('all_findings','moving_confinement','all_active_moves_blocked'))+
            ' | '+percent(rate(row['controls_matching'],row['controls']))+' |')
    lines += ['', f"There were **{findings['moving_without_perpendicular_faces_15']:,} moving cases without "
        'nearby perpendicular faces**, and '
        f"**{findings['moving_without_two_obstacles_15']:,} without two collision rectangles within 15 units**. "
        'The exact coordinates, clearances, headings and trace references of counterexamples '
        'are retained; a single corner-only explanation would miss these cases.', '',
        '## Geometry rules evaluated on separate maps', '',
        'Rules are chosen using 80% of maps and evaluated on the other 20% (seed divisible by five). '
        'Only geometry, step length and pose are used in these rules; detected movement cycles '
        'and future escape outcomes are excluded. Training scores weight each approximate spot equally. '
        'Control specificity measures rejection of the saved unflagged windows; it is not a '
        'population estimate of trap probability.', '',
        '| Rule | Held-out predator coverage | Held-out spot-weighted coverage | Held-out control specificity |',
        '| --- | ---: | ---: | ---: |']
    for rule in rules:
        stats = rule['held_out']
        lines.append(f"| {rule['label']} | {percent(stats['case_coverage'])} | "
            f"{percent(stats['spot_weighted_case_coverage'])} | {percent(stats['control_specificity'])} |")
    if not rules:
        lines.append('| Insufficient training/control data | n/a | n/a | n/a |')
    lines += ['', '## Files and reproducibility', '',
        '- [All measured windows](windows.csv): one row per finding/control, entry and late geometry, '
        'headings, step availability, biome, movement fractions and recurrence periods.',
        '- [All approximate spots](spots.csv): map coordinates, clustered predator IDs and representative traces.',
        '- [Every hypothesis](conditions.csv): coverage, counterexample counts and controls for each mechanism.',
        '- [Example counterexamples](counterexamples.csv): up to five per hypothesis and mechanism.',
        '- [Machine-readable report](analysis.json), [coverage validation](coverage.json) and [feature provenance](settings.json).', '',
        'Run `python survival-simulator/scripts/predator_stuck_spots.py --workers 8` from the repository root. '
        'It streams the existing archives without extracting them, caches completed games atomically, '
        'and resumes on rerun. Use `--report-only` to rebuild tables/reports from the cache.', '',
        '## Scope of the conclusion', '',
        'The data establishes observed conditions and a geometric sufficient condition for one '
        'blocking mechanism. It cannot prove a complete universal necessary-and-sufficient rule '
        'for every possible map, heading and approach. The controls are selected, their windows '
        'occur at game end, and the trap windows are selected by the detector. Approximated spot '
        'groups can merge neighboring paths. No observed escape before 600 seconds does not imply '
        'permanent trapping. Step endpoints are checked as in the engine; this is not swept-path '
        'collision or a continuous free-space pathfinding test.', '']
    if settings.get('entrance_version'):
        lines[2:2] = ['This run also includes [observed entrances and exit-route tests](entrance_report.md). '
                      'Use `--entrances` when rerunning this extension.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='report_complete', **findings)), flush=True)
    return summary
