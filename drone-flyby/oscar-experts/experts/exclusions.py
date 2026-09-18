"""Label exclusions reviewed by Oscar (data/drone/sprite-mask-review-20260918/label_exclusions.json).

An excluded label is neither a target nor background: tiles carrying one are skipped entirely.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'data/drone/sprite-mask-review-20260918/label_exclusions.json'


class Exclusions:
    def __init__(self, path=None):
        path = Path(path) if path else DEFAULT
        self.rules = json.loads(path.read_text()) if path.exists() else []

    def excluded(self, track_id, frame):
        return any(r['track_id'] == track_id and ('frames' not in r or frame in r['frames']) for r in self.rules)

    def tile_has_excluded(self, record):
        return any(self.excluded(a.get('track_id'), record.get('frame')) for a in record.get('annotations', []))
