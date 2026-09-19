"""Recovery contracts; no games, providers or LLM calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.research_io import atomic_json, directory_bytes
from scripts.optimize_policy import run_case, retryable_case_error
from scripts.research_resilient import validate_panels, ResilientSupervisor, failed_infrastructure
from scripts.research_support import read, write, validate_config
from scripts.research_loop import Supervisor
from tests.test_research_supervisor import configuration


class RecoveryContracts(unittest.TestCase):
    def test_storage_handles_temporary_file_disappearing(self):
        vanished = Mock()
        vanished.stat.side_effect = FileNotFoundError('renamed temporary file')
        existing = Mock()
        existing.stat.return_value = Mock(st_mode=0o100600, st_size=123)
        @contextmanager
        def scan(_):
            yield iter([vanished, existing])
        with patch('scripts.research_io.os.scandir', scan):
            self.assertEqual(directory_bytes('/example'), 123)

    def test_storage_does_not_hide_permission_errors(self):
        with patch('scripts.research_io.os.scandir', side_effect=PermissionError()):
            with self.assertRaises(PermissionError):
                directory_bytes('/example')

    def test_concurrent_writers_use_distinct_atomic_temporary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'progress.json'
            def writer(number):
                for _ in range(40):
                    atomic_json(target, {'writer':number, 'payload':[number]*100})
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(writer, range(4)))
            value = read(target)
            self.assertEqual(value['payload'], [value['writer']]*100)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_only_technical_case_failures_are_retried(self):
        self.assertTrue(retryable_case_error({'status':'error','error':'Wall-time limit reached'}))
        self.assertFalse(retryable_case_error({'status':'extinct','sim_time':1}))
        self.assertFalse(retryable_case_error({'status':'error','error':'ValueError: broken policy'}))

    def test_timeout_retry_doubles_wall_time_without_changing_case(self):
        request = dict(case_id='abc', infrastructure_retries=2)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            first = dict(case_id='abc',status='error',error='Wall-time limit reached')
            success = dict(case_id='abc',status='extinct',sim_time=1)
            with patch('scripts.optimize_policy._run_case_once', side_effect=[first,success]) as run:
                self.assertEqual(run_case(request,out,out/'control',100),success)
                self.assertEqual([c.args[3] for c in run.call_args_list],[100,200])
                self.assertTrue(all(c.args[0] is request for c in run.call_args_list))

    def test_successful_cases_are_never_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            result=dict(case_id='abc',status='extinct',sim_time=1)
            write(out/'cases/abc/result.json',result)
            with patch('scripts.optimize_policy._run_case_once') as run:
                self.assertEqual(run_case(dict(case_id='abc',infrastructure_retries=2),out,out,100),result)
                run.assert_not_called()

    def test_retry_cap_survives_coordinator_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            failure=dict(case_id='abc',status='error',error='Wall-time limit reached')
            for n in range(3):write(out/f'cases/abc/attempt-{n}/result.json',failure)
            with patch('scripts.optimize_policy._run_case_once') as run:
                self.assertEqual(run_case(dict(case_id='abc',infrastructure_retries=2),out,out,100),failure)
                run.assert_not_called()

    def test_seed_panels_exclude_retired_and_new_holdouts(self):
        cfg=configuration()
        cfg['evaluation'].update(holdout_seeds=[900],retired_holdout_seeds=[901],seed_panels={
            'g2.1':dict(screen_seeds=[20],comparison_seeds=[20,21],major_seeds=[20,21,22]),
            'g2.2':dict(screen_seeds=[30],comparison_seeds=[30,31],major_seeds=[30,31,32])})
        self.assertEqual(validate_panels(cfg),{20,21,22,30,31,32})
        for invalid in (900,901,20):
            broken=copy.deepcopy(cfg)
            broken['evaluation']['seed_panels']['g2.2']['major_seeds'].append(invalid)
            with self.assertRaises(ValueError):validate_panels(broken)

    def test_round_uses_frozen_panel_and_restores_base_config(self):
        with tempfile.TemporaryDirectory() as folder:
            supervisor=ResilientSupervisor.__new__(ResilientSupervisor)
            supervisor.out=Path(folder)
            original=configuration()
            panel=dict(screen_seeds=[20],comparison_seeds=[20,21],major_seeds=[20,21,22])
            original['evaluation']['seed_panels']={'g2.1':panel}
            supervisor.config=original
            def inner(*_):
                self.assertEqual(supervisor.config['evaluation']['screen_seeds'],[20])
                return {'improved':False}
            with patch.object(Supervisor,'round',side_effect=inner):
                supervisor.round('g2.1',{}, {})
            self.assertIs(supervisor.config,original)
            self.assertEqual(read(supervisor.out/'seed-exposure/g2.1.json'),panel)

    def test_reviewer_errors_require_technical_evidence(self):
        self.assertTrue(failed_infrastructure({'returncode':1},'stream disconnected'))
        self.assertFalse(failed_infrastructure({'returncode':1},'invalid output schema'))
        self.assertFalse(failed_infrastructure({'returncode':0,'status':'complete'},'old connection reset'))

    def test_independent_confirmation_is_disjoint_and_valid(self):
        cfg=configuration()
        cfg['evaluation'].update(independent_confirmation=True,screen_seeds=[20,21],
            comparison_seeds=[30,31],major_seeds=[30,31,32])
        cfg['budget']['agent_call_reserve_usd']=0
        validate_config(cfg)
        cfg['evaluation']['screen_seeds']=[30]
        with self.assertRaises(ValueError):validate_config(cfg)


if __name__=='__main__':unittest.main()
