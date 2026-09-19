"""Native integration contracts, with the real build checked when present."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.research_support import research_case, manifest, changed_policy, read, validate_config
from scripts.simulation_backend import identity

ROOT = Path(__file__).resolve().parents[1]


class NativeContracts(unittest.TestCase):
    def test_reference_engine_required_for_promotion_and_holdout(self):
        for name in ('comparison_engine', 'holdout_engine'):
            config = read(ROOT/'scripts/research_config.json')
            config['budget']['hourly_rate_usd'] = 1.
            config['evaluation'][name] = 'fastsim'
            with self.assertRaisesRegex(ValueError, 'reference Python'):
                validate_config(config)

    def test_native_sources_and_binary_are_frozen_but_bundle_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'fastsim'
            folder.mkdir()
            for name in ('_engine.cpp', '_engine.so', 'build-info.json'):
                (folder/name).write_text('original')
            before = manifest(tmp)
            portable = manifest(tmp, native=False)
            self.assertEqual(set(portable), {'fastsim/_engine.cpp'})
            (folder/'_engine.cpp').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Protected'):
                changed_policy(before, manifest(tmp))

    def test_policy_boundary_rejects_native_engine_import(self):
        from models.observation_only import _assert_no_simulator
        with patch.dict(sys.modules, {'fastsim': object()}):
            with self.assertRaisesRegex(RuntimeError, 'fastsim'):
                _assert_no_simulator()

    @unittest.skipUnless((ROOT/'fastsim/build-info.json').exists(), 'native engine not built on this host')
    def test_native_build_and_python_have_separate_cache_identity(self):
        identity('fastsim')
        args = ('snapshot', {}, {}, False, 0, 0)
        self.assertNotEqual(research_case(*args)['case_id'], research_case(*args, engine='fastsim')['case_id'])

    @unittest.skipUnless((ROOT/'fastsim/build-info.json').exists(), 'native engine not built on this host')
    def test_native_world_rng_and_accounting_match_normalized_python(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/verify_native_diagnostics.py'),
                                 '--steps', '300'], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report['ok'])
        self.assertGreater(report['totals']['births'], 0)
        self.assertGreater(sum(report['deaths'].values()), 0)


if __name__ == '__main__':
    unittest.main()
