"""Regression checks for discovery, immutable routing and concurrent publication."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from catalog import ReplayCatalog, replay_digest


def fixture(title='Run', created='2026-09-17T10:00:00Z'):
    return dict(format='survival-replay', version=1, meta=dict(title=title, created_at=created),
                world=dict(width=100, height=100, obstacles=[]), events=[],
                frames=[dict(t=0, score=0, agents=[], predators=[], fruits=[], trees=[]),
                        dict(t=1, score=1, agents=[], predators=[], fruits=[], trees=[])],
                summary=dict(duration=1, frames=2, score=1, reason='finished'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt') as stream:
        json.dump(data, stream)


class Discovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.catalog = ReplayCatalog(self.root)

    def test_frame_streaming_preserves_canonical_content_ids(self):
        data = fixture()
        data['frames'][0]['agents'] = [{'name': 'Fruit æ', 'energy': 0.123456789, 'alive': True}]
        payload = {'created_at': data['meta']['created_at'], 'world': data['world'],
                   'frames': data['frames'], 'events': data['events'], 'summary': data['summary']}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:32]
        self.assertEqual(replay_digest(payload), expected)

    def test_nested_replays_and_copies_are_available_once(self):
        data = fixture()
        original = self.root/'results/strategy/deep/a.json.gz'
        duplicate = self.root/'debugger/recordings/a.json'
        write(original, data)
        curated = copy.deepcopy(data);curated['meta']['title'] = 'Friendly label'
        write(duplicate, curated)
        write(self.root/'debugger/recordings/manifest.json', {'recordings': [dict(file='recordings/a.json', title='Preferred label')]})
        # A distinct run with the same title is not conflated with a copy.
        write(self.root/'research/elsewhere/b.json', fixture(created='later'))
        write(self.root/'results/metrics.json', {'score': 12})
        result = self.catalog.refresh()
        self.assertEqual(len(result['recordings']), 2)
        first = result['recordings'][0]
        self.assertEqual(first['title'], 'Preferred label')
        self.assertEqual(len(first['sources']), 2)
        self.assertEqual(self.catalog.resolve(first['file']), duplicate)

    def test_incomplete_file_becomes_available_after_completion(self):
        path = self.root/'results/live.json'
        path.parent.mkdir();path.write_text('{"format":"survival-replay",')
        self.assertEqual(self.catalog.refresh()['recordings'], [])
        write(path, fixture())
        entry = self.catalog.refresh()['recordings'][0]
        self.assertEqual(self.catalog.resolve(entry['file']), path)
        # A rewritten source must never be served under its previous content ID.
        changed = fixture();changed['frames'][-1]['score'] = 2
        write(path, changed)
        self.assertIsNone(self.catalog.resolve(entry['file']))
        updated = self.catalog.refresh()['recordings'][0]
        self.assertNotEqual(updated['file'], entry['file'])
        path.unlink()
        self.assertEqual(self.catalog.refresh()['recordings'], [])

    def test_excludes_outside_symlinks_and_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)/'secret.json';write(target, fixture())
            (self.root/'escape.json').symlink_to(target)
            bad = fixture();bad['summary']['duration'] = 10
            write(self.root/'truncated.json', bad)
            write(self.root/'vendor/test.json', fixture())
            self.assertEqual(self.catalog.refresh()['recordings'], [])
            self.assertIsNone(self.catalog.resolve('replays/../../secret.json'))

    def test_cache_does_not_reparse_unchanged_replays(self):
        from unittest.mock import patch
        write(self.root/'one.json', fixture())
        self.catalog.refresh()
        with patch('catalog.read_replay', side_effect=AssertionError('unnecessary reparse')):
            self.assertEqual(len(self.catalog.refresh()['recordings']), 1)


if __name__ == '__main__':
    unittest.main()
