"""Merge collected shards only after complete, unique seed coverage is verified."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def merge(root, games=10000):
    counts=Counter();groups=defaultdict(Counter);examples=defaultdict(list)
    seeds=set();sources=None;config=None;shards=[];input_reports=[]
    reports=sorted(root.glob('shard-*-reports'))
    for report in reports:
        marker=root/(report.name.removesuffix('-reports')+'-verified.json')
        if not marker.is_file():
            raise ValueError(f'Archive download not yet verified: {report.name}')
        summary=json.loads((report/'summary.json').read_text())
        manifest=json.loads((report/'manifest.json').read_text())
        analysis=json.loads((report/'diagnostic_analysis.json').read_text())
        parity=json.loads((report/'parity-linux.json').read_text())
        assert summary['failed_games']==0 and summary['completed_games']==summary['requested_games']
        assert parity['status']=='passed'
        if sources is None:
            sources=manifest['sources']
            config=manifest['config']
        assert sources==manifest['sources'], 'Mixed source versions'
        assert config==manifest['config']==summary['config'], 'Mixed configurations'
        expected=set(range(manifest['start_seed'],manifest['start_seed']+summary['requested_games']))
        actual=set(analysis['seeds'])
        assert actual==expected and len(actual)==len(analysis['seeds']), 'Shard seed coverage mismatch'
        assert not seeds.intersection(actual), 'Duplicate seeds across shards'
        seeds.update(actual);counts.update(analysis['counts'])
        counts['failed_games']+=summary['failed_games']
        for key,value in analysis['groups'].items():
            for name,number in value.items():
                # Older shard reports summed this per-predator maximum. Keep
                # the value, but name the aggregate explicitly rather than
                # presenting it as one predator's longest blocked interval.
                if name=='longest_blocked_active_run':
                    name='sum_of_predator_longest_blocked_active_runs'
                groups[key][name]+=number
        for key,value in analysis['examples'].items():
            examples[key].extend(dict(shard=report.name,**v) for v in value)
        shards.append(json.loads(marker.read_text()))
        input_reports.append(dict(shard=report.name,
            analysis_sha256=hashlib.sha256((report/'diagnostic_analysis.json').read_bytes()).hexdigest()))
    assert seeds==set(range(games)),f'Only {len(seeds)}/{games} unique seeds collected'
    assert counts['completed_games']==games and counts['initial_predators']==100*games
    temporary=root/'diagnostic_findings.csv.tmp'
    with temporary.open('w',newline='',encoding='utf-8') as out:
        writer=None;rows=0
        for report in reports:
            with (report/'diagnostic_findings.csv').open(newline='',encoding='utf-8') as f:
                reader=csv.DictReader(f)
                if writer is None:
                    writer=csv.DictWriter(out,fieldnames=['shard',*reader.fieldnames]);writer.writeheader()
                for row in reader:
                    writer.writerow(dict(shard=report.name,**row));rows+=1
    assert rows==counts['flagged_predators']
    temporary.replace(root/'diagnostic_findings.csv')
    result=dict(status='complete',counts=dict(counts),groups={k:dict(v) for k,v in groups.items()},
        examples={k:v[:12] for k,v in examples.items()},shards=shards,source_hashes=sources,
        config=config,input_reports=input_reports,
        local_analysis_source_sha256=hashlib.sha256(Path(__file__).with_name('analyze.py').read_bytes()).hexdigest(),
        merge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coverage=dict(unique_seeds=len(seeds),first_seed=min(seeds),last_seed=max(seeds),duplicates=0),
        limitations='Only the first qualifying 60-second interval per predator is counted. Follow-up is censored at 600 seconds. Controls are descriptive; replay interventions support conclusions for specific cases, not universal necessary/sufficient rules.')
    (root/'diagnostic_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    merge(args.root)
