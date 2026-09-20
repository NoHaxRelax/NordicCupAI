"""Independent CPython candidate check, positive fixture, and optional broad parity."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--baseline', type=Path)
p.add_argument('--samples', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()

def scan(binary, samples, start, end):
    return subprocess.run([str(binary.resolve()), str(samples), str(start), str(end)],
                          capture_output=True, check=True).stdout

with tempfile.TemporaryDirectory(prefix='scan-check-') as tmp:
    weak = Path(tmp)/'samples.txt'
    weak.write_text('200 200 0\n')
    expected = []
    for seed in range(16387):
        r = random.Random(seed)
        points = [(r.randrange(1600), r.randrange(1200)) for _ in range(10)]
        labels = [r.randrange(4) for _ in range(10)]
        nearest = min(range(10), key=lambda i: (points[i][0]-200)**2+(points[i][1]-200)**2)
        if labels[nearest] == 0:
            expected.append(seed)
    assert list(map(int, scan(a.binary, weak, 0, 16387).split())) == expected
    assert list(map(int, scan(a.binary, a.samples, 1894581200, 1894581400).split())) == [1894581302]
    result = dict(passed=True, independent_seeds=16387, independent_candidates=len(expected), positive_fixture=True)
    if a.baseline:
        old = scan(a.baseline, weak, 0, 4194304)
        new = scan(a.binary, weak, 0, 4194304)
        assert old == new
        result.update(broad_seeds=4194304, broad_candidates=len(new.splitlines()),
                      candidate_sha256=hashlib.sha256(new).hexdigest())
    result['binary_sha256'] = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
