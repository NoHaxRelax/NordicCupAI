"""One isolated full-horizon evaluation. Launched by optimize_policy.py.

Top-level imports are standard-library-only so Windows spawn does not load the
simulator in the policy process. Scores and spectator state stay in this evaluator.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    from scripts.research_io import atomic_json
    atomic_json(path, data)


def evaluate(request, output, stop_file):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['SDL_VIDEODRIVER'] = os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    sys.path.insert(0, str(ROOT))
    from models.experiment_actor import ExperimentActor
    from scripts.simulation_backend import create, identity, observations
    from src.utils.DTOs import ActionRequest

    started, cpu = time.monotonic(), time.process_time()
    result = dict(case_id=request['case_id'], seed=request['seed'], mode=request['mode'],
                  seconds=request['seconds'], policy_seed=request['policy_seed'])
    diagnostics, sim, actor = None, None, None
    samples = []
    try:
        # Allowlist, not a formality: every listed mode runs the unmodified engine
        # with natural predator spawning. Nothing here disables spawning or forces
        # an encounter, and an unknown mode is rejected rather than assumed safe.
        if request['mode'] not in ('trapping', 'orchard_evasion', 'expert_harvest'):
            raise ValueError('Only natural-predator games are supported by this optimizer')
        engine = request.get('engine', 'python')
        engine_build = identity(engine)
        if request.get('engine_build', engine_build) != engine_build:
            raise ValueError('Requested engine build differs from the executing build')
        result.update(engine=engine, engine_build=engine_build)
        sim = create(engine, seed=request['seed'])
        if request.get('diagnostics', {}).get('enabled', False):
            from scripts.research_diagnostics import Diagnostics
            diagnostics = Diagnostics(sim.env, output.parent/'diagnostics', request, sim.dt)
        states = observations(sim, engine)
        peak, ids = len(states), {s['agent_id'] for s in states}
        limit = round(request['seconds']/sim.dt)
        next_progress, next_stop_check, status = 0., 0., 'horizon'
        samples = []
        policy_rpc_wall, max_policy_rpc_wall, agent_decisions = 0., 0., 0
        with ExperimentActor(request['mode'], request['config'], request['baseline'], request['policy_seed'],
                             diagnostics=diagnostics is not None) as actor:
            for tick in range(limit):
                if not states:
                    status = 'extinct'
                    break
                wall_now = time.monotonic()
                if wall_now >= next_stop_check:
                    heartbeat = stop_file.parent/'heartbeat'
                    abandoned = not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime > 90.
                    if stop_file.exists() or abandoned:
                        status = 'interrupted'
                        break
                    next_stop_check = wall_now+.5
                if diagnostics:
                    diagnostics.before(tick, states)
                call_started = time.perf_counter()
                decisions = actor(states, sim.env.time)
                call_seconds = time.perf_counter()-call_started
                policy_rpc_wall += call_seconds
                max_policy_rpc_wall = max(max_policy_rpc_wall, call_seconds)
                agent_decisions += len(states)
                if diagnostics:
                    diagnostics.decision(decisions, actor.last_audit)
                state = sim.step([(aid, ActionRequest.model_validate(a)) for aid, a in decisions])
                if diagnostics:
                    diagnostics.after()
                elif engine == 'fastsim':
                    sim.pop_events()
                states = state['observations']
                peak = max(peak, len(states))
                ids.update(s['agent_id'] for s in states)
                elapsed = time.monotonic()-started
                if elapsed >= next_progress:
                    sample = dict(sim_time=sim.env.time, horizon=request['seconds'],
                                  score=sim.env.score, alive=len(states), wall_seconds=elapsed)
                    samples.append(sample)
                    write_json(output.parent/'progress.json', dict(result, **sample))
                    next_progress = elapsed+10.
            if not states:
                status = 'extinct'
            result.update(status=status, sim_time=sim.env.time, score=sim.env.score,
                          final_agents=len(states), peak_agents=peak, total_agents=len(ids),
                          final_predators=len(sim.env.predators), predators_enabled=True, audit=actor.last_audit,
                          policy_rpc_wall_seconds=policy_rpc_wall, max_policy_rpc_wall_seconds=max_policy_rpc_wall,
                          agent_decisions=agent_decisions,
                          policy_rpc_seconds_per_1000_agent_decisions=1000*policy_rpc_wall/max(1, agent_decisions),
                          samples=samples)
    except KeyboardInterrupt:
        result.update(status='interrupted')
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    if sim is not None:
        result.update(sim_time=sim.env.time, score=sim.env.score, final_agents=len(sim.env.agents), samples=samples)
    if actor is not None:
        result['audit'] = actor.last_audit
    if diagnostics is not None:
        try:
            result['diagnostics'] = diagnostics.finish(result['status'])
        except Exception:
            result.update(status='error', diagnostic_error=traceback.format_exc())
    result.update(wall_seconds=time.monotonic()-started, evaluator_cpu_seconds=time.process_time()-cpu)
    write_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args()
    evaluate(json.loads(args.request.read_text(encoding='utf-8')), args.output, args.stop_file)


if __name__ == '__main__':
    main()
