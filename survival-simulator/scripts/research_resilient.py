"""Resumable research with fixed seed panels and bounded infrastructure retries.

Run only against a newly frozen recovery campaign. Historical campaigns and
their final selections remain immutable. No retry depends on a survival score.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_loop import Supervisor, NoBudget, NoStorage, Paused
from scripts.research_support import read, write, remaining_seconds
from scripts.optimize_policy import study_lock


def validate_panels(config):
    settings = config['evaluation']
    forbidden = set(settings['holdout_seeds']) | set(settings.get('retired_holdout_seeds', []))
    exposed = set()
    for name, panel in settings.get('seed_panels', {}).items():
        groups = [panel[k] for k in ('screen_seeds', 'comparison_seeds', 'major_seeds')]
        if any(not g or len(set(g)) != len(g) or any(type(s) is not int or s < 0 for s in g) for g in groups):
            raise ValueError('Invalid seed panel: '+name)
        a, b, c = map(set, groups)
        nested = (not a & c and b <= c) if settings.get('independent_confirmation') else a <= b <= c
        if not nested or (a | b | c) & (forbidden | exposed):
            raise ValueError('Seed panel overlaps another round or reserved holdout: '+name)
        exposed.update(a | b | c)
    return exposed


def failed_infrastructure(result, log=''):
    if result.get('returncode') == 0 and result.get('status') == 'complete':
        return False
    text = (str(result.get('error', ''))+' '+log).lower()
    return any(s in text for s in ('connection reset', 'connection closed', 'transport error',
        'temporarily unavailable', 'rate limit', 'too many requests', 'stream disconnected',
        'wrapper missing', 'fileNotFoundError'.lower(), 'no such file or directory'))


class ResilientSupervisor(Supervisor):
    def __init__(self, out):
        super().__init__(out)
        validate_panels(self.config)

    def round(self, key, incumbent, upstream, major=False):
        original = self.config
        panel = original['evaluation'].get('seed_panels', {}).get(key)
        self.config = copy.deepcopy(original)
        if panel:
            self.config['evaluation'].update(panel)
            path = self.out/'seed-exposure'/f'{key}.json'
            if path.exists() and read(path) != panel:
                raise ValueError('Frozen round seed panel changed')
            write(path, panel)
        try:
            return super().round(key, incumbent, upstream, major)
        finally:
            self.config = original

    def execute(self, key, seconds, builder, **kwargs):
        limit = self.config.get('recovery', {}).get('step_retries', 2)
        for _ in range(limit+1):
            result = super().execute(key, seconds, builder, **kwargs)
            folder = self.out/'steps'/key
            logpath = folder/'console.log'
            log = logpath.read_text(encoding='utf-8', errors='replace')[-24000:] if logpath.exists() else ''
            if not failed_infrastructure(result, log):
                return result
            retries = self.state.setdefault('recovery_attempts', {}).get(key, 0)
            left = remaining_seconds(self.config, self.state, time.time(), final=kwargs.get('final', False))
            if retries >= limit or left < min(seconds, self.config['schedule']['minimum_step_seconds']):
                return result
            archive = folder/'recovery'/str(retries+1)
            archive.mkdir(parents=True, exist_ok=False)
            for name in ('process-result.json', 'process-request.json', 'console.log'):
                source = folder/name
                if source.exists():
                    source.replace(archive/name)
            # A completed wrapper has reaped its own child group before it writes
            # process-result.json. New attempts receive separate control paths.
            control = folder/'control'
            if control.exists():
                control.replace(archive/'control')
            control.mkdir()
            self.state['steps'][key].update(status='pending', deadline=time.time()+min(seconds, left))
            self.state['recovery_attempts'][key] = retries+1
            self.event(f'{key}.recovery-{retries+1}', 'infrastructure retry',
                       dict(previous_result=result, archive=str(archive)))
            time.sleep(min(30, 5*(retries+1)))
        return result


def run(out):
    """Restart a failed coordinator; preserve case results and all candidate IDs."""
    config = read(out/'config.json')
    retries = config.get('recovery', {}).get('coordinator_retries', 5)
    for _ in range(retries+1):
        try:
            with study_lock(out/'supervisor.lock'):
                supervisor = ResilientSupervisor(out)
                preflight = out/'recovery-preflight.json'
                if not preflight.exists():
                    for ref in (supervisor.state['original'], supervisor.state['best']):
                        if not supervisor.checks(ref['snapshot'], 'recovery-checks'):
                            raise RuntimeError('Recovered candidate failed trusted checks')
                    write(preflight, dict(complete=True, at=time.time(),
                        original=supervisor.state['original']['id'], best=supervisor.state['best']['id']))
                supervisor.run()
                return
        except (OSError, TimeoutError) as exc:
            state = read(out/'state.json')
            count = state.get('coordinator_retries', 0)
            if count >= retries or (out/'STOP').exists():
                raise
            state.update(coordinator_retries=count+1, status='recovering', error=None)
            write(out/'state.json', state)
            write(out/'recovery-errors'/f'coordinator-{count+1}.json',
                  dict(error_type=type(exc).__name__, error=str(exc), at=time.time()))
            time.sleep(min(30, 5*(count+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
