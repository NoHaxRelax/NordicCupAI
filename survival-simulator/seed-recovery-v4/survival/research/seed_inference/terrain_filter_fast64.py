#!/usr/bin/env python3
"""Adapt Seed spoof's fast64 scanner to the streaming verifier CLI."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

samples, start, count, workers, output = sys.argv[1:]
start, count, workers = int(start), int(count), int(workers)
scanner = Path(os.environ.get('SEED_FAST64_BINARY', str(Path(__file__).with_name('scan-fast-64'))))
began = time.monotonic()
chunk = 2**20
def scan(lo):
    hi = min(start + count, lo + chunk)
    run = subprocess.run([str(scanner), samples, str(lo), str(hi)],
                         capture_output=True, text=True, check=True)
    return hi - lo, run.stdout

checked = hits = 0
with open(output, 'w') as out, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(scan, lo) for lo in range(start, start + count, chunk)]
    for future in concurrent.futures.as_completed(futures):
        n, candidates = future.result()
        out.write(candidates)
        out.flush()
        checked += n
        hits += len(candidates.split())
        print(json.dumps(dict(tested=checked, hits=hits, seconds=time.monotonic()-began)), flush=True)
