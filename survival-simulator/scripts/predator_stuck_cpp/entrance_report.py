"""Report observed walk-in confinement separately from geometric impossibility."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CASE_FIELDS = ['seed', 'predator_id', 'spot_id', 'classification', 'biome',
    'center_x', 'center_y', 'entry_seconds', 'crossing_from_x', 'crossing_from_y',
    'crossing_to_x', 'crossing_to_y', 'entry_segment_clear', 'detected_seconds',
    'observed_residence_seconds', 'followup_seconds', 'entered_no_exit_with_60s_followup',
    'exit_test', 'exit_route_steps', 'exit_search_expanded', 'entered_behavioral_trap',
    'entered_geometric_sink', 'entered_exit_unresolved', 'spawned_overlapping',
    'pose_period', 'late_nearest_clearance', 'late_second_clearance', 'shard', 'trace']
FLAGS = ['observed_legal_entry', 'entered_no_observed_exit', 'entered_then_escaped',
    'entered_no_exit_with_60s_followup', 'entered_behavioral_trap', 'entered_geometric_sink',
    'entered_exit_unresolved']


def make_entrance_report(output):
    output = Path(output)
    coverage = json.loads((output/'coverage.json').read_text())
    settings = json.loads((output/'settings.json').read_text())
    counts = Counter(); evidence = Counter(); exit_tests = Counter()
    mechanisms = {}; examples = {}; durations = []
    maps_with_entry = set(); maps_with_trap = set(); maps_unknown = set()
    with (output/'windows.csv').open(newline='', encoding='utf-8') as f, \
         (output/'entry_traps.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=CASE_FIELDS); writer.writeheader()
        for row in csv.DictReader(f):
            if row['is_case'] != '1': continue
            counts['findings'] += 1
            evidence[row['entry_evidence']] += 1
            exit_tests[row['exit_test']] += 1
            group = mechanisms.setdefault(row['classification'], Counter())
            for key in FLAGS:
                counts[key] += int(row[key]); group[key] += int(row[key])
            seed = int(row['seed'])
            if row['observed_legal_entry'] == '1':
                if row['entered_then_escaped'] == '1' and row['no_legal_first_step_proven'] == '1':
                    raise ValueError('Observed escape contradicts a no-first-step proof')
                maps_with_entry.add(seed)
                counts['entries_with_clear_crossing_segment'] += row['entry_segment_clear']=='1'
                counts['entries_with_segment_clipping'] += row['entry_segment_clear']=='0'
            else:
                maps_unknown.add(seed)
            if row['entered_no_observed_exit'] == '1':
                maps_with_trap.add(seed)
                durations.append(float(row['observed_residence_seconds']))
                writer.writerow({key:row[key] for key in CASE_FIELDS})
                counts['no_exit_with_clear_crossing_segment'] += row['entry_segment_clear']=='1'
            key = row['exit_test']
            if len(examples.setdefault(key, [])) < 5:
                examples[key].append({k:row[k] for k in CASE_FIELDS})
    if counts['findings'] != coverage['cases']:
        raise ValueError('Entry analysis does not cover every finding')
    if counts['entered_no_observed_exit'] != sum(counts[k] for k in (
            'entered_behavioral_trap','entered_geometric_sink','entered_exit_unresolved')):
        raise ValueError('Entry/exit classifications do not partition the candidate cases')
    spots_total = 0; candidate_spots = 0; multi_entry_spots = 0; geometric_spots = 0; unresolved_spots = 0
    with (output/'spots.csv').open(newline='', encoding='utf-8') as f, \
         (output/'entry_spots.csv').open('w', newline='', encoding='utf-8') as out:
        reader = csv.DictReader(f); writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            spots_total += 1
            if int(row['entered_no_observed_exit']) == 0: continue
            candidate_spots += 1; writer.writerow(row)
            multi_entry_spots += int(row['entered_no_observed_exit']) >= 2
            geometric_spots += int(row['entered_geometric_sink']) > 0
            unresolved_spots += int(row['entered_exit_unresolved']) > 0
    counts.update(maps_with_observed_entry=len(maps_with_entry), maps_with_entered_no_exit=len(maps_with_trap),
        approximate_spots=spots_total, entry_trap_spots=candidate_spots,
        spots_with_multiple_entered_no_exit_predators=multi_entry_spots,
        spots_with_proven_geometric_sink=geometric_spots, spots_with_unresolved_exit=unresolved_spots)
    summary = dict(status='complete', counts=dict(counts), entry_evidence=dict(evidence),
        exit_tests=dict(exit_tests), mechanisms={k:dict(v) for k,v in mechanisms.items()},
        residence_seconds_quantiles=dict(zip(('minimum','median','p95','maximum'),
            map(float,np.quantile(durations,[0,.5,.95,1])))) if durations else {},
        examples=examples, coverage=coverage, settings=settings,
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantics=dict(entry='Accepted, unclamped native move crossing from outside the original 15-unit circle to inside, with both endpoints collision-free, followed by at least 60 seconds continuously inside.',
            policy_confinement='Observed entry, no departure before game end, and a constructive exit path under arbitrary headings. This does not mean the native policy will select the path.',
            geometric_sink='At detection, all headings are blocked at the native biome-scaled step length, established by partitioning the entire step circle against obstacle rectangles. A sufficient no-first-step condition under this static, no-agent walking model.',
            unresolved='The bounded heading/position search failed. This is not proof that no exit exists.',
            witnesses='Per-game feature cache stores searched paths, or the trace/entry tick needed to reconstruct a verified reverse route. Route lengths checked to 1e-7 units; collision clearance must be at least 1e-8.',
            limitations='Only first detected confinement per predator and up to 30 seconds of approach were saved. Entry may therefore be unknown. Follow-up is censored at game end. Native collision uses endpoints, so segment clipping is counted separately. Controls are not an inventory of all visits or failed capture attempts.'))
    (output/'entrance_analysis.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    rows = [
        ('Confinement findings checked', counts['findings']),
        ('Observed legal entries followed by at least 60 seconds inside', counts['observed_legal_entry']),
        ('Entered and no departure observed before game end', counts['entered_no_observed_exit']),
        ('Entered, then later left the circle', counts['entered_then_escaped']),
        ('No observed departure, with at least 60 seconds of additional follow-up after detection', counts['entered_no_exit_with_60s_followup']),
        ('No observed departure, but a legal exit route was verified', counts['entered_behavioral_trap']),
        ('No observed departure and no legal first step exists at any heading', counts['entered_geometric_sink']),
        ('No observed departure; exit search unresolved', counts['entered_exit_unresolved']),
        ('Approximate spots with at least one entered/no-departure predator', counts['entry_trap_spots']),
        ('Those spots reached by at least two such predators', counts['spots_with_multiple_entered_no_exit_predators'])]
    lines = ['# Places predators enter and then remain confined', '',
        f"Checked all **{counts['findings']:,} confinement findings from {coverage['games']:,} games**. "
        f"There are **{counts['entry_trap_spots']:,} approximate spots** where a recorded legal entry "
        'was followed by confinement and no observed departure through game end.', '',
        '| Measurement | Count |', '| --- | ---: |']
    lines.extend(f'| {label} | {value:,} |' for label,value in rows)
    lines += ['', '## What qualifies as entering', '',
        'The saved approach must show the predator stepping from outside its original 15-unit '
        'test circle to inside it. Both step endpoints must be collision-free, the native engine '
        'must have accepted the move without bounds clamping, and at least 60 seconds inside '
        'must follow. Being spawned inside a trap does not qualify by itself. A predator that '
        'spawned overlapping earlier can qualify if it subsequently made a recorded legal entry.', '',
        f"Of the observed entries, {counts['entries_with_clear_crossing_segment']:,} also have a "
        f"clear straight crossing segment; {counts['entries_with_segment_clipping']:,} clip an "
        'expanded obstacle between their legal endpoints. The latter are native-legal steps '
        'because this simulator does not check collisions along the whole segment. '
        f"The stricter clear-crossing subset contains {counts['no_exit_with_clear_crossing_segment']:,} "
        'entered/no-departure cases.', '',
        '## What the exit tests establish', '',
        '**A verified exit route establishes that geometry permits escape with different '
        'heading choices.** First the analyzer tries to reverse the recorded approach and '
        'confinement moves, checking the biome-scaled step length at every reversed position. '
        'Biome transitions can make a forward move impossible to reverse exactly. If reversal '
        'fails, it searches for another route to outside the same test circle. Actual route '
        'coordinates are not rounded; every saved route is checked again for legal endpoints '
        'and correct step lengths. Energy/rest pauses do not change this static geometric route test.', '',
        '**A proven geometric sink has no legal first step for any heading.** The test partitions '
        'the complete step circle at every rectangle intersection, then checks the intervals '
        'and their boundaries with a positive collision margin. It is stronger than checking '
        'a finite heading grid. The conclusion is for the fixed native walking step at that '
        'position, in this static no-agent experiment.', '',
        '**Unresolved cases remain unresolved.** The fallback search uses '
        f"{settings['escape_angle_step']:g}-degree directions, additional narrow-opening samples, "
        f"a 0.5-unit visitation grid and a {settings['escape_max_nodes']}-node budget. "
        'Failure to find a route under those limits does not establish impossibility. '
        'Likewise, no observed departure by 600 seconds does not establish that the normal '
        'predator policy can never escape.', '',
        '## Locations and evidence', '',
        '- [Candidate locations](entry_spots.csv): approximate spot coordinates and counts of entry/no-exit cases.',
        '- [Every entered/no-departure case](entry_traps.csv): entry coordinates/time, residence, mechanism and exit-test result.',
        '- [Full entry/exit analysis](entrance_analysis.json): coverage, evidence categories and example cases.',
        '- [All windows](windows.csv): includes later escape and missing-entry evidence, not only candidate traps.',
        '- `game-features/seed-NNNNNNNN.json.gz`: exit witnesses, including full searched paths or references for reversed recorded paths.', '',
        'Run from the repository root:', '',
        '```powershell',
        '& "..\\NordicCupAI\\survival-simulator\\.venv\\Scripts\\python.exe" survival-simulator/scripts/predator_stuck_spots.py --entrances --workers 8',
        '```', '',
        'Use `--entrances --report-only` to rebuild reports from completed caches. '
        'The entrance mode preserves the original spot analysis in its own directory. '
        'Use a new output directory when increasing the search budget or changing its resolution.', '',
        'Missing entry evidence is not evidence of impossible entry: the saved approach is limited '
        'to 30 seconds. Counts also exclude captures shorter than the original 60-second detector. '
        'The location groups are the existing five-unit connected components of late path centers; '
        'they are approximate and are not precise trap boundaries.', '']
    (output/'entrance_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='entrance_report_complete', **counts)), flush=True)
    return summary
