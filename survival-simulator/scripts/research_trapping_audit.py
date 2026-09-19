"""Read-only comparison of deployed trapping sources with the pinned Lucas mirror."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

CAMPAIGN = Path('/workspace/research-night-1')
MIRROR = Path('/workspace/lucas-bridge')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()


def main():
    state, upstream = read(CAMPAIGN / 'state.json'), read(MIRROR / 'current.json')
    original = Path(upstream['source'])
    files = ['models/core.py'] + sorted(p.relative_to(original).as_posix()
                                       for p in (original / 'models/entrapment').glob('*.py'))
    roots = {'initial': CAMPAIGN / 'snapshots' / state['initial_snapshot'] / 'code',
             'incumbent': CAMPAIGN / 'snapshots' / state['best']['snapshot'] / 'code'}
    for name in ('guide-delivery', 'bait-continuity', 'capture-allocation'):
        roots[name] = Path(read(CAMPAIGN.parent / 'research-capture-fleet' / name / 'plan.json')['source'])
    sources = {}
    for label, root in roots.items():
        manifest = read(root.parent / 'manifest.json')
        rows = []
        for name in files:
            path = root / name
            rows.append(dict(file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                text_sha256=fingerprint(path), matches_lucas_text=fingerprint(path) == fingerprint(original / name),
                matches_frozen_manifest=hashlib.sha256(path.read_bytes()).hexdigest() == manifest.get(name)))
        sources[label] = dict(root=str(root), files=rows)
    checks = subprocess.run(['/workspace/predator-search-venv/bin/python', '-B', '-m', 'unittest',
        'tests.test_current_integration', 'tests.test_current_guide_delivery', 'tests.test_current_guide_steering'],
        cwd=roots['incumbent'], text=True, capture_output=True, timeout=60)
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), upstream_commit=upstream['commit'],
        github_checked_at=upstream['github_checked_at'], sources=sources,
        tests=dict(returncode=checks.returncode, output=(checks.stdout + checks.stderr)[-4000:]),
        all_frozen_files_verified=all(r['matches_frozen_manifest'] for value in sources.values() for r in value['files']))
    print(json.dumps(report, indent=2))
    if checks.returncode or not report['all_frozen_files_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
