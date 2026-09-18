"""Local 1,000-map overview and on-demand native replays from frozen source.

python scripts/guide_batch_viewer.py logs/guide_batch/runpod-20260918-1000
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--port', type=int, default=9057)
    args = parser.parse_args()
    root = args.folder.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    multi = bool(manifest['config'].get('multi'))
    run_glob = 'multi-*' if multi else 'map-*'
    for name, expected_hash in manifest['source_hashes'].items():
        if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError(f'Frozen source changed: {name}')
    case_file = root/'cases.json'
    if not case_file.exists():
        case_file = root/'results.json'
    cases = {r['index']:r for r in json.loads(case_file.read_text())}
    aggregate_file = root/'aggregate.json'
    if aggregate_file.exists():
        aggregate = json.loads(aggregate_file.read_text())
    else:
        passed = sum(r['outcome']=='delivery_pass' for r in cases.values())
        eligible = sum(r['outcome']!='no_usable_bait_site' for r in cases.values())
        aggregate = dict(maps=len(cases),
                         success_all_maps=dict(successes=passed,total=len(cases),
                                               fraction=passed/max(1,len(cases))),
                         success_known_eligible_maps=dict(successes=passed,total=eligible,
                                                          fraction=passed/max(1,eligible)),
                         wrong_side_at_final_frame=sum(r['outcome']=='wrong_side' for r in cases.values()),
                         wrong_side_final_period=sum(bool(r.get('replacement_side_final_period'))
                                                     for r in cases.values()))
    jobs, lock = {}, threading.Lock()
    pool = ThreadPoolExecutor(max_workers=2)

    def status(index):
        folder = root/'replays'/f'case-{index:04}'
        with lock:
            pending = dict(jobs.get(index, {}))
        if pending.get('state') in ('queued','rendering','failed'):
            pending['frames'] = sum(1 for _ in folder.glob(run_glob+'/frames/*.png'))
            return pending
        summaries = sorted(folder.glob(run_glob+'/summary.json'))
        for p in summaries:
            summary = json.loads(p.read_text())
            if multi and any(summary.get(k)!=cases[index].get(k) for k in ('outcome','seconds','final')):
                continue
            if summary.get('frames',0) and (p.parent/'frames/0.png').exists():
                return dict(state='ready', url='/'+str(p.parent.relative_to(root))+'/index.html',
                            frames=summary['frames'], outcome=summary['outcome'])
        with lock:
            state = dict(jobs.get(index, dict(state='not_generated')))
        state['frames'] = sum(1 for _ in folder.glob(run_glob+'/frames/*.png'))
        return state

    def render(index):
        folder = root/'replays'/f'case-{index:04}'
        folder.mkdir(parents=True, exist_ok=True)
        with lock:
            jobs[index] = dict(state='rendering')
        case = cases[index]
        command = [sys.executable, str(root/'source/scripts/guide_lab.py'),
                   '--seed', str(case['seed']), '--encounter-seed', str(case['encounter_seed']),
                   '--seconds', str(manifest['config']['seconds']), '--width', '640',
                   '--output', str(folder)]
        if multi:
            command = [sys.executable,str(root/'source/scripts/guide_multi.py'),'--deliveries','1',
                       '--seed',str(case['seed']),'--encounter-seed',str(case['encounter_seed']),
                       '--output',str(folder)]
            for option in ('replace_bait','vision_delivery'):
                if manifest['config'].get(option):
                    command.append('--'+option.replace('_','-'))
        env = os.environ | {'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        try:
            with (folder/'render.log').open('w') as output:
                result = subprocess.run(command, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=300 if multi else 180)
            summaries = sorted(folder.glob(run_glob+'/summary.json'))
            if result.returncode or not summaries:
                raise RuntimeError('Replay process failed; see render.log.')
            summary = json.loads(summaries[0].read_text())
            if summary.get('error') or not summary.get('frames'):
                raise RuntimeError(summary.get('error','No frames were produced.'))
            if multi and any(summary.get(k)!=case.get(k) for k in ('outcome','seconds','final')):
                raise RuntimeError('Rerender differs from recorded evaluation; use the original tick trace.')
            with lock:
                jobs[index] = dict(state='ready')
        except subprocess.TimeoutExpired:
            with lock:
                jobs[index] = dict(state='failed', message='Replay exceeded its rendering time limit. The recorded batch trace remains available.')
        except Exception as error:
            with lock:
                jobs[index] = dict(state='failed', message=str(error))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, payload, code=200):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path=='/':
                html=Path(__file__).with_name('guide_multi_viewer.html') if multi else Path(__file__).with_suffix('.html')
                data = html.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path=='/api/cases':
                compact=[]
                for index,r in sorted(cases.items()):
                    item={k:r.get(k) for k in ('index','seed','encounter_seed','outcome','seconds','frames',
                          'final','guide_caught','eligible_sites','contact_metrics','site','error',
                          'outcome_rear_at_end_only','deliveries','initial_min_held','final_hold_min','replacement_side_final_period')}
                    trace=next((root/'retries'/f'case-{index:04}').glob('map-*/ticks.jsonl.gz'),None)
                    if trace is None:
                        trace=next((root/f'case-{index:04}').glob(run_glob+'/ticks.jsonl.gz'),None)
                    item['trace']='/'+str(trace.relative_to(root)) if trace else None
                    item['replay']=status(index)
                    compact.append(item)
                self.send_json(dict(summary=aggregate,cases=compact))
            elif path=='/aggregate.json':
                self.send_json(aggregate)
            elif path=='/cases.json':
                self.send_json(list(cases.values()))
            elif path=='/protocol.txt' and not (root/'protocol.txt').exists():
                data = ('Frozen evaluation configuration:\n'+
                        json.dumps(manifest['config'],indent=2)+
                        '\n\nReplays rerun the frozen source with the recorded seeds.\n'
                        'Multi-predator outcomes and final state must match before display.\n').encode()
                self.send_response(200)
                self.send_header('Content-Type','text/plain; charset=utf-8')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path.startswith('/api/replay/'):
                try:
                    index=int(path.rsplit('/',1)[1])
                    if index not in cases: raise ValueError()
                    self.send_json(status(index))
                except ValueError:
                    self.send_json(dict(error='Unknown case'),404)
            else:
                super().do_GET()

        def do_POST(self):
            # Only same-origin browser actions can schedule local replay work.
            origin=self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                self.send_json(dict(error='Origin rejected'),403)
                return
            path=urlsplit(self.path).path
            try:
                if not path.startswith('/api/replay/'): raise ValueError()
                index=int(path.rsplit('/',1)[1])
                if index not in cases: raise ValueError()
            except ValueError:
                self.send_json(dict(error='Unknown case'),404)
                return
            if cases[index]['outcome']=='no_usable_bait_site':
                self.send_json(dict(state='unavailable', message='This map has no qualifying bait site.'),422)
                return
            current=status(index)
            if current['state'] in ('not_generated','failed'):
                with lock:
                    queued=sum(j['state'] in ('queued','rendering') for j in jobs.values())
                    if queued>=4:
                        self.send_json(dict(error='Four replays are already queued; wait for one to finish.'),429)
                        return
                    if jobs.get(index,{}).get('state') not in ('queued','rendering'):
                        jobs[index]=dict(state='queued')
                        pool.submit(render,index)
                current=status(index)
            self.send_json(current)

    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(root)))
    print(f'Batch overview: http://127.0.0.1:{args.port}/',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pool.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':
    main()
