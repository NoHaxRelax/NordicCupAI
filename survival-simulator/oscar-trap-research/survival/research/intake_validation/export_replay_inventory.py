"""Inventory original local recordings; omit reproducible viewer chunks."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
SCOPES = ('depth5_test', 'intake_gate_sol', 'intake_geometry_sol',
          'intake_guides_sol', 'intake_validation', 'real_map_intake_sol',
          'wall_funneling')


def main():
    rows = []
    for scope in SCOPES:
        for path in sorted((ROOT / 'survival/results' / scope).rglob('*.json.gz')):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            rows.append({'path': str(path.relative_to(ROOT)),
                         'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    output = {'schema': 'local-replay-inventory-v1',
              'availability': 'Local only; recordings are excluded from Git. This inventory is not a download service.',
              'count': len(rows), 'total_bytes': sum(row['bytes'] for row in rows),
              'files': rows}
    (ROOT / 'LOCAL_REPLAY_INVENTORY.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f"Inventoried {len(rows)} recordings, {output['total_bytes']} bytes")


if __name__ == '__main__':
    main()
