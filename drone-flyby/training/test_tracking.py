"""CPU-only tracking checks; uses synthetic data and never contacts W&B online."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

from tracking import Tracker, scalars


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='drone-tracking-test-')
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def tracker(self, **kwargs):
        return Tracker(output=self.output, project='nordic-ai-cup-drone', **kwargs)

    def sdk(self):
        run = Mock(id='synthetic-run', dir=str(self.output/'files'), url='https://example.invalid/run')
        return SimpleNamespace(Settings=Mock(), init=Mock(return_value=run), Artifact=Mock()), run

    def test_disabled_needs_no_sdk(self):
        with patch.dict('sys.modules', {'wandb': None}):
            tracker = self.tracker(mode='disabled')
            tracker.log({'train': {'loss': 1}}, 1)
            tracker.finish({'status': 'completed'})
        state = json.loads((self.output/'tracking.json').read_text())
        self.assertEqual(state['status'], 'disabled')
        self.assertEqual(state['training_status'], 'completed')

    def test_online_requires_explicit_workspace(self):
        with self.assertRaisesRegex(ValueError, 'explicit'):
            self.tracker(mode='online')

    def test_online_initialization_can_fall_back_to_offline(self):
        sdk, run = self.sdk()
        sdk.init.side_effect = [ConnectionError('do not record this secret'), run]
        with patch.dict('sys.modules', {'wandb': sdk}), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tracker = self.tracker(mode='online', entity='synthetic-workspace')
        self.assertEqual([call.kwargs['mode'] for call in sdk.init.call_args_list], ['online', 'offline'])
        self.assertEqual(tracker.state['mode'], 'offline')
        self.assertNotIn('secret', (self.output/'tracking-errors.jsonl').read_text())

    def test_provider_outage_does_not_abort_epoch_or_finish(self):
        sdk, run = self.sdk()
        run.log.side_effect = ConnectionError('sensitive provider text')
        run.finish.side_effect = RuntimeError('provider unavailable')
        with patch.dict('sys.modules', {'wandb': sdk}), warnings.catch_warnings():
            warnings.simplefilter('error')
            tracker = self.tracker(mode='offline')
            tracker.log({'train': {'loss': 0.5}}, 1)
            tracker.finish(error_type='InterruptedError')
        self.assertEqual(tracker.state['errors'], 2)
        self.assertEqual(tracker.state['training_status'], 'failed')
        run.finish.assert_called_once_with(exit_code=1)

    def test_full_initialization_outage_keeps_local_receipt(self):
        sdk, _ = self.sdk()
        sdk.init.side_effect = ConnectionError()
        with patch.dict('sys.modules', {'wandb': sdk}), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tracker = self.tracker(mode='online', entity='synthetic-workspace')
            tracker.log({'train': {'loss': 0.5}}, 1)
            tracker.finish({'status': 'completed'})
        self.assertEqual(tracker.state['status'], 'unavailable')
        self.assertEqual(tracker.state['training_status'], 'completed')

    def test_checkpoint_upload_is_explicit(self):
        checkpoint = self.output/'last.pt'
        checkpoint.write_bytes(b'synthetic checkpoint, not a model')
        for enabled in (False, True):
            with self.subTest(upload_checkpoints=enabled):
                sdk, _ = self.sdk()
                with patch.dict('sys.modules', {'wandb': sdk}):
                    tracker = self.tracker(mode='offline', upload_checkpoints=enabled)
                    tracker.finish({'checkpoint': str(checkpoint)})
                kinds = [call.kwargs['type'] for call in sdk.Artifact.call_args_list]
                self.assertEqual('model' in kinds, enabled)

    def test_metrics_keep_zoom_and_evaluation_namespaces(self):
        metrics = scalars({'holdout': {'L1': {'accuracy': .75, 'confusion_matrix': [[1, 0], [1, 2]]}},
                           'train_fit': {'mAP50': .9}, 'bad': float('nan')})
        self.assertEqual(metrics, {'holdout/L1/accuracy': .75, 'train_fit/mAP50': .9})

    def test_real_offline_sdk_persists_history_and_artifacts(self):
        manifest = self.output/'manifest.json'
        manifest.write_text(json.dumps({'synthetic': True, 'classes': ['example']}))
        tracker = self.tracker(mode='offline', manifest=manifest,
                               config={'task': 'tracking-smoke', 'synthetic': True})
        self.assertIsNotNone(tracker.run)
        tracker.config({'training': {'epochs': 2}})
        tracker.log({'train': {'loss': .8}, 'holdout': {'L1': {'accuracy': .5}}}, 1)
        tracker.log({'train': {'loss': .6}, 'holdout': {'L1': {'accuracy': .75}}}, 2)
        summary = dict(tracker.run.summary)
        self.assertEqual(summary['epoch'], 2)
        self.assertEqual(summary['holdout/L1/accuracy'], .75)
        tracker.finish({'status': 'completed', 'synthetic': True})
        state = json.loads((self.output/'tracking.json').read_text())
        self.assertEqual(state['errors'], 0)
        self.assertEqual(state['mode'], 'offline')
        logs = list((self.output/'wandb').glob('offline-run-*/run-*.wandb'))
        self.assertEqual(len(logs), 1)
        # The closed transaction file must contain the actual scalar keys and
        # metadata artifacts. No reliance on removed SDK-internal readers.
        persisted = logs[0].read_bytes()
        for value in (b'holdout/L1/accuracy', b'train/loss', b'manifest.json', b'run-metadata', b'dataset'):
            self.assertIn(value, persisted)
        self.assertNotIn(b'final-model', persisted)


if __name__ == '__main__':
    unittest.main()
