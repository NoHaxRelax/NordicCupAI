"""Spawned JSON-only policy boundary for baseline and experimental comparisons."""
import multiprocessing
import os
import sys
import time

from models.observation_only import (
    ObservationOnlyOrchard, TIMEOUT_SECONDS, ACTION_FIELDS, AGENT_FIELDS,
    _assert_no_simulator, _encode, _decode, sanitize_states, _number,
)


def _worker(connection):
    try:
        # Inherited evaluation CLI arguments include a world seed. Do not expose
        # them to policy modules; the startup payload contains only policy settings.
        sys.argv = ['experiment-policy-worker']
        os.environ.pop('EXPERT_POLICY_CONFIG', None)
        os.environ.pop('GLOBAL_PLANNER_CONFIG', None)
        _assert_no_simulator()
        if not connection.poll(TIMEOUT_SECONDS):
            raise TimeoutError('No startup request')
        startup = _decode(connection.recv_bytes())
        from src.utils.DTOs import ObservationResponse
        if set(ObservationResponse.model_fields) != set(AGENT_FIELDS):
            raise RuntimeError('Observation whitelist differs from official schema')
        mode, seed = startup['mode'], startup['policy_seed']
        seed_used = True
        if mode == 'trapping':
            if startup['baseline']:
                from models.core import EntrapmentPolicy
                policy = EntrapmentPolicy(seed=seed)
            else:
                from models.experimental_policy import ExperimentalEntrapment
                policy = ExperimentalEntrapment(startup['config'], seed=seed)
        elif mode == 'orchard_evasion':
            from models.orchard_evasion_policy import OrchardEvasionPolicy
            if startup['baseline']:
                policy = OrchardEvasionPolicy(seed=seed)
            else:
                from models.notrap_config import orchard_kwargs
                policy = OrchardEvasionPolicy(seed=seed, **orchard_kwargs(startup['config']))
        elif mode == 'expert_harvest':
            # This stack carries no policy RNG, so the case's policy seed cannot
            # apply; the audit records that rather than implying seeded variation.
            from models.optimization_policy import OptimizationPolicy
            seed_used = False
            if startup['baseline']:
                policy = OptimizationPolicy()
            else:
                policy = OptimizationPolicy(startup['config']['expert'], startup['config']['planner'])
        else:
            raise ValueError('Unknown optimization mode')
        diagnostics = startup.get('diagnostics', False)
        policy.record_decisions = diagnostics
        _assert_no_simulator()
        audit = dict(env_loaded=False, transport='json', process_start='spawn',
                     policy_seed=seed, policy_seed_used=seed_used, mode=mode, baseline=startup['baseline'],
                     policy_class=f'{type(policy).__module__}.{type(policy).__name__}')
        connection.send_bytes(_encode(dict(kind='ready', audit=audit)))
        decision_calls, decision_cpu_seconds, decision_wall_seconds = 0, 0., 0.
        while True:
            request = _decode(connection.recv_bytes())
            if request['kind'] == 'close':
                break
            if request['kind'] != 'step':
                raise ValueError('Unknown worker request')
            states = sanitize_states(request['states'])
            now = _number(request['sim_time'], 'sim_time', minimum=0)
            _assert_no_simulator()
            cpu_started, wall_started = time.process_time(), time.perf_counter()
            actions = policy(states, now)
            decision_cpu_seconds += time.process_time() - cpu_started
            decision_wall_seconds += time.perf_counter() - wall_started
            decision_calls += 1
            _assert_no_simulator()
            encoded, seen = [], set()
            ids = {s['agent_id'] for s in states}
            for aid, action in actions:
                if aid not in ids or aid in seen or aid != action.agent_id:
                    raise ValueError('Invalid or duplicate action ID')
                seen.add(aid)
                encoded.append([aid, {key: getattr(action, key) for key in ACTION_FIELDS}])
            debug, debug_seconds = {}, 0.
            if diagnostics:
                debug_started = time.perf_counter()
                from models.diagnostic_export import export
                debug = export(policy)
                if len(_encode(debug)) > 512*1024:
                    debug = dict(roles=debug.get('roles', {}), truncated=True)
                debug_seconds = time.perf_counter()-debug_started
            connection.send_bytes(_encode(dict(kind='actions', actions=encoded,
                audit=dict(env_loaded=False, metrics=getattr(policy, 'extra_metrics', {}),
                           decision_calls=decision_calls, decision_cpu_seconds=decision_cpu_seconds,
                           decision_wall_seconds=decision_wall_seconds,
                           trapping_metrics=getattr(policy, 'metrics', {}),
                           policy_debug=debug, debug_export_wall_seconds=debug_seconds))))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send_bytes(_encode(dict(kind='error', error=f'{type(exc).__name__}: {exc}')))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class ExperimentActor(ObservationOnlyOrchard):
    """Reuse the maintained sanitization, timeout and process-cleanup protocol."""
    def __init__(self, mode, config, baseline=False, policy_seed=0, diagnostics=False):
        context = multiprocessing.get_context('spawn')
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(target=_worker, args=(child,), daemon=True)
        self._closed = False
        self.audit, self.last_audit = {}, {}
        try:
            self._process.start()
            child.close()
            self._connection.send_bytes(_encode(dict(mode=mode, config=config,
                baseline=baseline, policy_seed=policy_seed, diagnostics=diagnostics)))
            self.audit = self._receive('ready')['audit']
            self.last_audit = dict(self.audit)
        except BaseException:
            child.close()
            self.close()
            raise
