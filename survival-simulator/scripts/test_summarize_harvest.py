"""Reject invalid headline cohorts before publishing benchmark results."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_harvest import combine


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def shard(self, name, start=1, n=1000, **changes):
        counters = ('attempts confirmed failed actual_harvests first_tick_harvests '
                    'failed_farms sleeping_targets early_wakes false_ignores ignored_sightings '
                    'retries sacrifices cap_rejections floor_rejections skip_rejections').split()
        row = dict.fromkeys(counters, 0)
        row.update(engine='native', arm='candidate', horizon=3000., sim_time=3000.,
                   score=1., population=6, peak_actions=6)
        row.update(changes)
        path = Path(self.directory.name)/name
        path.write_text(''.join(json.dumps(dict(row, seed=i))+'\n' for i in range(start, start+n)))
        manifest = dict(arguments=dict(seeds=n, engine='native', arm='candidate',
                                      horizon=3000., extra_drain_actions=0),
                        source_sha256={'policy.py':'same-source'})
        path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
        return path

    def test_accepts_exactly_1000_complete_unique_games(self):
        path = self.shard('complete.jsonl')
        rows, totals, _ = combine([path])
        self.assertEqual(len(rows), 1000)
        self.assertEqual(totals['n'], 1000)
        self.assertEqual(totals['full_horizon'], 1000)

    def test_rejects_999_even_when_shard_itself_finished(self):
        with self.assertRaisesRegex(ValueError, '1000 distinct'):
            combine([self.shard('short.jsonl', n=999)])

    def test_rejects_overlapping_shards(self):
        paths = [self.shard('one.jsonl', n=500), self.shard('two.jsonl', n=500)]
        with self.assertRaisesRegex(ValueError, '1000 distinct'):
            combine(paths)

    def test_rejects_live_unfinished_game(self):
        with self.assertRaisesRegex(ValueError, 'incomplete game'):
            combine([self.shard('unfinished.jsonl', sim_time=2000.)])

    def test_accepts_game_ended_by_extinction(self):
        _, totals, _ = combine([self.shard('extinct.jsonl', sim_time=2000., population=0)])
        self.assertEqual(totals['full_horizon'], 0)

    def test_rejects_mixed_policy_sources(self):
        paths = [self.shard('one.jsonl', n=500), self.shard('two.jsonl', start=501, n=500)]
        manifest = paths[1].with_suffix('.manifest.json')
        data = json.loads(manifest.read_text())
        data['source_sha256']['policy.py'] = 'different-source'
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'mixed source'):
            combine(paths)


if __name__ == '__main__':
    unittest.main()
