"""Optional W&B tracking with explicit destinations and local failure receipts."""
import json
import math
from numbers import Real
from pathlib import Path
import time
import warnings


def scalars(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(scalars(child, f'{prefix}/{key}' if prefix else str(key)))
    elif isinstance(value, Real) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


class Tracker:
    def __init__(self, *, output, mode, project, entity=None, group=None,
                 config=None, manifest=None, upload_checkpoints=False):
        if mode not in ('disabled', 'offline', 'online'):
            raise ValueError('Unknown tracking mode')
        if mode == 'online' and not entity:
            raise ValueError('Online tracking requires an explicit --wandb-entity or WANDB_ENTITY')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.run = None
        self.upload_checkpoints = upload_checkpoints
        self.state = dict(requested_mode=mode, mode=mode, project=project, entity=entity,
                          status='disabled' if mode == 'disabled' else 'starting', errors=0,
                          upload_checkpoints=upload_checkpoints)
        self._record()
        if mode == 'disabled':
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('Install requirements-tracking.txt before enabling W&B') from exc
        self.wandb = wandb
        settings = wandb.Settings(console='off', disable_git=True, disable_code=True,
                                  init_timeout=30, x_disable_meta=True)
        kwargs = dict(project=project, entity=entity, name=self.output.name, group=group,
                      job_type=(config or {}).get('task', 'training'), config=config or {},
                      dir=str(self.output), mode=mode, settings=settings, save_code=False)
        try:
            self.run = wandb.init(**kwargs)
        except Exception as exc:
            self._error('init', exc)
            if mode == 'online':
                try:
                    kwargs['mode'] = 'offline'
                    self.run = wandb.init(**kwargs)
                    self.state['mode'] = 'offline'
                except Exception as fallback:
                    self._error('offline_fallback', fallback)
        if self.run is not None:
            self.state.update(status='active', run_id=self.run.id, run_dir=self.run.dir,
                              url=self.run.url if self.state['mode']=='online' else None)
            self._call('define_metrics', lambda: self.run.define_metric('*', step_metric='epoch'))
            if manifest:
                self.file_artifact('dataset-manifest', 'dataset', [manifest])
        else:
            self.state['status'] = 'unavailable'
        self._record()

    def _record(self):
        temporary = self.output/'tracking.tmp'
        temporary.write_text(json.dumps(self.state, indent=2)+'\n')
        temporary.replace(self.output/'tracking.json')

    def _error(self, operation, exc):
        self.state['errors'] += 1
        # Do not persist credentials or arbitrary provider exception text.
        row = dict(operation=operation, error_type=type(exc).__name__, time=time.time())
        try:
            with (self.output/'tracking-errors.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            self._record()
        except OSError:
            pass
        # A warnings-as-errors policy must not turn a tracking outage into a
        # training failure either.
        try:
            warnings.warn(f'W&B {operation} failed ({type(exc).__name__}); local training continues.', stacklevel=2)
        except Warning:
            pass

    def _call(self, operation, function):
        if self.run is None:
            return
        try:
            return function()
        except Exception as exc:
            self._error(operation, exc)

    def config(self, values):
        self._call('config', lambda: self.run.config.update(values, allow_val_change=False))

    def log(self, values, epoch):
        payload = dict(scalars(values), epoch=epoch)
        self._call('log', lambda: self.run.log(payload))

    def file_artifact(self, name, kind, paths):
        def upload():
            paths_existing = [Path(p) for p in paths if Path(p).is_file()]
            if not paths_existing:
                return
            artifact = self.wandb.Artifact(f'{name}-{self.run.id}', type=kind)
            for path in paths_existing:
                artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
        self._call('artifact', upload)

    def finish(self, result=None, error_type=None):
        result = result or {}
        summary = dict(result, status='failed' if error_type else result.get('status', 'completed'))
        if error_type:
            summary['error_type'] = error_type
        self.state['training_status'] = summary['status']
        try:
            self._record()
        except OSError:
            pass
        self._call('summary', lambda: self.run.summary.update(summary))
        self.file_artifact('run-receipt', 'run-metadata',
                           [self.output/p for p in ('runtime.json','result.json','progress.json','history.json','tracking.json')])
        if self.upload_checkpoints and result.get('checkpoint'):
            self.file_artifact('final-model', 'model', [result['checkpoint']])
        self._call('finish', lambda: self.run.finish(exit_code=1 if error_type else 0))
        if self.run is not None:
            self.state['status'] = 'finished'
        try:
            self._record()
        except OSError:
            pass
