"""Local water deployment experiments; no network or engine modifications."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'water_deployment'
sys.path.insert(0, str(ROOT / 'vendor' / 'survival-simulator'))
sys.path.insert(0, str(ROOT / 'research'))
SOURCE = 'acfc31a4003a5f91bf11032a02cd98c178ddbd7e'
EXPERIMENT_HASHES={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(Path(__file__).parent.glob('*.py'))}

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def save(name, data):
    OUT.mkdir(exist_ok=True)
    path = OUT / (name + '.json')
    metadata = dict(source_commit=SOURCE, engine_sha256={
        str(p.relative_to(ROOT / 'vendor')): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / 'vendor' / 'survival-simulator').rglob('*.py'))},
        experiment_sha256=EXPERIMENT_HASHES)
    path.write_text(json.dumps(dict(metadata=metadata, data=data), indent=2,
        allow_nan=False, default=lambda x: x.item()) + '\n')
    print(path, flush=True)
