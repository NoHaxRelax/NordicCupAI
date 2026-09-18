"""Frozen-source screen of delivery radius and walking threshold on paired cases.

Development screening only; evaluate finalists on independent maps before
claiming general reliability. Production policies are never edited by this tool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-variants', type=int, default=2)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = {str(p.relative_to(ROOT)): p.read_bytes()
              for directory in ('models','src','scripts') for p in (ROOT/directory).rglob('*')
              if p.is_file() and p.suffix in ('.py','.json','.html')}
    guide_path = 'models/entrapment/my_guide.py'
    original = source[guide_path].decode()
    stop_pattern = 'math.hypot(*bait) <= HEARING_TARGET'
    speed_pattern = "predator['distance'] < 100 else agent['speed']"
    assert stop_pattern in original and speed_pattern in original

    def run(config):
        radius, walk = config
        folder = args.output/f'delivery{radius}-walk{walk}'
        snapshot = folder/'source'
        for name, contents in source.items():
            target = snapshot/name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == guide_path:
                contents = original.replace(stop_pattern, f'math.hypot(*bait) <= {radius}.0').replace(
                    speed_pattern, f"predator['distance'] < {walk} else agent['speed']").encode()
            target.write_bytes(contents)
        command = [sys.executable,str(snapshot/'scripts/guide_batch.py'),'--multi','--maps','100',
                   '--shards','5','--shard','0','--workers',str(args.workers),'--timeout','300',
                   '--output',str(folder/'batch')]
        with (folder/'worker.log').open('w') as handle:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
        cases = json.loads((folder/'batch/results.json').read_text())
        row = dict(delivery_radius=radius,walking_threshold=walk,maps=len(cases),
                   passed=sum(r['outcome']=='delivery_pass' for r in cases),
                   outcomes={s:sum(r['outcome']==s for r in cases) for s in {r['outcome'] for r in cases}})
        (folder/'screen.json').write_text(json.dumps(row,indent=2)+'\n')
        return row

    configs = list(itertools.product((45,50,55),(85,100,115)))
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel_variants) as pool:
        for future in as_completed([pool.submit(run,c) for c in configs]):
            row = future.result()
            results.append(row)
            print(json.dumps(row),flush=True)
            (args.output/'screen.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__': main()
