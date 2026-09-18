"""Build an explicit-allowlist bundle for a Runpod CPU pod: vendored simulator, society harness,
orchard policy and sweep runner. Never includes .git, .gpu, private/, .env, credentials or results."""
import pathlib, tarfile, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]          # survival/
OUT = ROOT.parent / 'artifacts' / 'orchard-cpu-bundle.tgz'
FILES = [
    'vendor/survival-simulator/src', 'vendor/survival-simulator/requirements.txt',
    'research/society/harness.py', 'research/simple_policies.py',
    'research/orchard/orchard.py', 'research/orchard/harness_np.py', 'research/orchard/sweep.py',
]
OUT.parent.mkdir(exist_ok=True)
with tarfile.open(OUT, 'w:gz') as tar:
    for rel in FILES:
        p = ROOT / rel
        assert p.exists(), rel
        tar.add(p, arcname=f'survival/{rel}', filter=lambda ti: None if '__pycache__' in ti.name or ti.name.endswith('.pyc') else ti)
print(OUT, OUT.stat().st_size // 1024, 'KB')
