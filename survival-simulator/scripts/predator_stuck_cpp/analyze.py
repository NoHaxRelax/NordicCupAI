"""Summarize completed diagnostic games, including unflagged population controls."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

from diagnostics import summarize


def analyze(folder):
    counts=Counter()
    groups=defaultdict(Counter)
    examples=defaultdict(list)
    seeds=[]
    fields=['seed','predator_id','biome','start_seconds','detected_seconds','spawned_overlapping_obstacle',
        'initial_overlap_at_interval','classification','active_ticks','all_candidates_blocked_ticks',
        'successful_fallback_ticks','quarter_turn_ticks','four_active_step_returns','turn_sign_changes',
        'travelled_distance','escaped_after_detection','first_exit_seconds','followup_seconds',
        'nearest_collision_clearance','second_collision_clearance','third_collision_clearance','trace']
    temporary=folder/'diagnostic_findings.csv.tmp'
    with temporary.open('w',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for path in sorted((folder/'games').glob('seed-*/result.json')):
            game=json.loads(path.read_text())
            if game['status']!='complete':
                counts['unfinished_games']+=1
                continue
            counts['completed_games']+=1
            counts['games_with_findings']+=bool(game['findings'])
            seeds.append(game['seed'])
            flagged={e['predator_id'] for e in game['findings']}
            for p in game['predators']:
                eligible=game['simulated_seconds']-p['born_seconds'] >= 60
                counts['total_predators']+=1
                counts['eligible_predators']+=eligible
                counts['initial_predators']+=p['born_seconds']==0
                key='spawn_overlap' if p['spawned_overlapping_obstacle'] else 'legal_spawn'
                g=groups[key]
                g['predators']+=1;g['eligible']+=eligible;g['flagged']+=p['predator_id'] in flagged
                motion=p.get('motion_totals',{})
                assert motion['rest_ticks']+motion['active_ticks']==round((game['simulated_seconds']-p['born_seconds'])/.1)
                for name,value in motion.items():
                    if name == 'longest_blocked_active_run':
                        name = 'sum_of_predator_longest_blocked_active_runs'
                    groups['flagged_lifetime' if p['predator_id'] in flagged else 'unflagged_lifetime'][name]+=value
            for e in game['findings']:
                d=e['diagnostic_summary']
                if 'quarter_turn_ticks' not in d:
                    with gzip.open(folder/e['trace'],'rt',encoding='utf-8') as f:
                        trace=json.load(f)
                    d=summarize(trace['diagnostic_rows'],trace['samples'][0][0])
                a=d['active_ticks'];b=d['all_candidates_blocked_ticks']
                assert e['sample_count']==601 and e['duration_seconds']==60
                assert e['max_distance']<=15+1e-9 and d['interval_ticks']==600
                assert (folder/e['trace']).is_file()
                kind='all_active_moves_blocked' if a and a==b else 'mixed_blocking' if b else 'moving_confinement'
                overlap=bool(e['onset_geometry']['overlapping_obstacles'])
                escaped=e['follow_up']['escaped_after_detection']
                quarter=a>0 and d['quarter_turn_ticks']/a>=.95
                cycle=a>4 and d['zero_displacement_active_ticks']<.05*a and d['four_active_step_returns']/(a-4)>=.90
                for key,inc in [('flagged_predators',1),(kind,1),('overlap_at_interval_start',overlap),
                    ('spawned_overlapping_and_flagged',e['spawned_overlapping_obstacle']),
                    ('quarter_turn_dominated',quarter),('four_step_cycle_dominated',cycle),
                    ('escaped_after_detection',escaped),('no_observed_escape',not escaped),
                    ('at_least_60s_followup',game['simulated_seconds']-e['detected_seconds']>=60)]:
                    counts[key]+=inc
                for key in ('biome:'+e['biome'],'mechanism:'+kind):
                    g=groups[key];g['flagged']+=1;g['quarter_turn_dominated']+=quarter
                    g['four_step_cycle_dominated']+=cycle;g['escaped']+=escaped;g['overlap_at_onset']+=overlap
                if len(examples[kind])<8:
                    examples[kind].append(dict(seed=e['seed'],predator_id=e['predator_id'],trace=e['trace']))
                nearest=e['onset_geometry']['nearest_obstacles']
                row={key:e[key] for key in ('seed','predator_id','biome','start_seconds','detected_seconds','spawned_overlapping_obstacle','trace')}
                row.update({key:d[key] for key in ('active_ticks','all_candidates_blocked_ticks','successful_fallback_ticks',
                    'quarter_turn_ticks','four_active_step_returns','turn_sign_changes','travelled_distance')})
                row.update(initial_overlap_at_interval=overlap,classification=kind,escaped_after_detection=escaped,
                    first_exit_seconds=e['follow_up']['first_exit_seconds'],followup_seconds=game['simulated_seconds']-e['detected_seconds'])
                for name,n in [('nearest_collision_clearance',0),('second_collision_clearance',1),('third_collision_clearance',2)]:
                    row[name]=nearest[n]['signed_collision_clearance']
                writer.writerow(row)
            if game.get('control_trace'):
                with gzip.open(folder/game['control_trace'],'rt',encoding='utf-8') as f:
                    control=json.load(f)
                rows=control['diagnostic_rows']
                d=summarize(rows,rows[-1][0]-600)
                g=groups['unflagged_control_windows'];g['windows']+=1
                g['quarter_turn_dominated']+=d['active_ticks']>0 and d['quarter_turn_ticks']/d['active_ticks']>=.95
                g['four_step_cycle_dominated']+=d['active_ticks']>4 and d['zero_displacement_active_ticks']<.05*d['active_ticks'] and d['four_active_step_returns']/(d['active_ticks']-4)>=.90
    temporary.replace(folder/'diagnostic_findings.csv')
    report=dict(counts=dict(counts),groups={k:dict(v) for k,v in groups.items()},examples=dict(examples),seeds=seeds,
        classification='Based on the first 60-second interval per predator. Moving confinement includes loops and oscillations. All-active-blocked means every active tick rejected all candidate moves.',
        limitations='No observed escape is censored at 600 seconds. Controls are the lowest-index unflagged predator per game and are descriptive, not randomized matched controls. Associations are not universal necessary/sufficient conditions.')
    temporary=folder/'diagnostic_analysis.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(folder/'diagnostic_analysis.json')
    print(json.dumps(counts),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    analyze(args.folder)
