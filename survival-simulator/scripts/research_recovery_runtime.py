"""External coordinator adapter; keep the campaign's frozen evaluator intact.

Its hash and operator handoff receipt must be saved before launch. Only repairs
fallback parameter selection and review scheduling; it never changes a policy,
score, case identity, budget, simulator or final holdout.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def fallback_paths(requested, inventory, limit):
    selectable = {s['path'] for s in inventory
                  if s.get('tunable') and not s['path'].startswith('features.')}
    result = list(dict.fromkeys(p for p in requested if p in selectable))[:limit]
    if not result:
        raise ValueError('No valid fallback tuning dimensions remain')
    return result


def fleet_budget(base, control):
    if control.get('total_usd') != 40 or control.get('reserve_usd') != 5 or control.get('auxiliary_allowance_usd') != 8:
        raise ValueError('Unexpected fleet budget authorization')
    result = copy.deepcopy(base)
    b = result['budget']
    if b['total_usd'] != 40 or b['reserve_usd'] != 5 or b['external_spend_usd'] != 10:
        raise ValueError('Unexpected original campaign budget')
    b['external_spend_usd'] += 8  # Reserve the full auxiliary ceiling up front.
    b['max_hours'] = min(b['max_hours'], 17.25)
    if b['external_spend_usd']+b['reserve_usd']+b['max_hours']*b['hourly_rate_usd'] > 40:
        raise ValueError('Fleet exceeds total authorized budget')
    return result


def throughput_budget(base, control):
    if (control.get('authorization') not in ('continue-four-pods-through-generation-8', 'scale-research-fleet-through-generation-8')
            or control.get('max_pods') not in (4, 8, 12)):
        raise ValueError('Missing throughput authorization')
    import math
    rate = control['max_pods']*.96+.11
    ceiling = math.ceil((15+72*rate)/100)*100
    result = copy.deepcopy(base)
    result['budget'].update(total_usd=ceiling, reserve_usd=5, external_spend_usd=10,
                            hourly_rate_usd=rate, max_hours=72)
    result['storage'].update(campaign_gib=180, final_reserve_gib=30)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fleet-controls', type=Path)
    parser.add_argument('--throughput-controls', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from scripts import research_resilient as runtime
    from scripts.research_support import read, write
    if args.throughput_controls:
        from scripts import research_loop
        from research_dispatch import remote_case
        controls = read(args.throughput_controls)
        queue = Path(controls['queue'])
        def dispatched_case(request, cache, control, timeout, *, runtime_root):
            return remote_case(queue, request, cache, control, timeout,
                               runtime_root=runtime_root, priority=0 if 'final-holdout' in str(control) else 5)
        research_loop.run_case = dispatched_case

    class OperationalSupervisor(runtime.ResilientSupervisor):
        def __init__(self, out):
            super().__init__(out)
            if args.throughput_controls:
                if controls['protocol_sha256'] != hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest():
                    raise ValueError('Throughput control belongs to another protocol')
                self.config = throughput_budget(self.config, controls)
                self.workers = 64  # Submission slots; actual CPU slots are owned by the shared workers.
                self.event('throughput-scale-20260919' if controls['max_pods'] > 4 else 'throughput-20260919', 'authorized shared fleet scheduling',
                    dict(controls=str(args.throughput_controls), previous_budget_usd=40,
                         expanded_spending_authorized=True, generation_target=8, pods=controls['max_pods'],
                         frozen_evaluator_unchanged=True, final_holdout_unchanged=True))
            if args.fleet_controls:
                control = read(args.fleet_controls)
                if control['protocol_sha256'] != hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest():
                    raise ValueError('Fleet control belongs to another protocol')
                self.config = fleet_budget(self.config, control)

        def save(self):
            super().save()
            if args.throughput_controls:
                path = self.out/'budget.json'
                budget = read(path)
                budget.update(previous_planning_target_usd=40, expandable_spending=True,
                    fleet_accounting_included=True, emergency_runtime_hours=72,
                    deployed_pods=controls['max_pods'], compute_hourly_usd=.96*controls['max_pods'],
                    note='Fleet calendar-time estimate. $40 is no longer a stopping limit. Stop at generation 8/final completion; 72h infrastructure failsafe.')
                write(path, budget)
            if args.fleet_controls:
                path = self.out/'budget.json'
                budget = read(path)
                budget.update(auxiliary_reserved_usd=8, historical_spend_usd=10,
                              note='Conservative committed cost: includes $8 reserved for three auxiliary Pods, plus historical spending; not a provider invoice.')
                write(path, budget)

        def search(self, key, snapshot, config, plan, major):
            field = 'broad_paths' if major else 'focus_paths'
            defaults = self.config['search']['default_paths']
            # The base major-review fallback explicitly copies default_paths.
            # Other explicit agent choices still receive strict validation.
            if not plan.get(field) or plan[field] == defaults:
                catalog = self.catalog(snapshot, key+'.fallback-catalog')
                selected = fallback_paths(defaults, catalog['inventory'],
                                          self.config['search']['max_dimensions'])
                plan = copy.deepcopy(plan)
                plan[field] = selected
                self.event(key+'.fallback-dimensions', 'validated fallback tuning dimensions',
                           dict(selected=selected, excluded=[p for p in defaults if p not in selected]))
            return super().search(key, snapshot, config, plan, major)

        def execute(self, key, seconds, builder, **kwargs):
            if args.throughput_controls:
                original_builder = builder
                def builder(folder, deadline):
                    command = original_builder(folder, deadline)
                    index = next((i for i, value in enumerate(command) if str(value).endswith('/scripts/research_study.py')), None)
                    if index is None or read(folder/'request.json').get('mode') == 'catalog':
                        return command
                    request = read(folder/'request.json')
                    request['workers'] = 24  # Keep BO adaptive instead of proposing every trial at once.
                    write(folder/'request.json', request)
                    return [sys.executable, '-B', str(Path(__file__).with_name('research_async_study.py')),
                        '--source', str(Path(command[index]).parents[1]), '--queue', str(queue),
                        *command[index+1:]]
            if kwargs.get('agent'):
                # The underlying executor clips this to the remaining funded
                # development budget and preserves the final reserve.
                seconds = max(seconds, 1800)
                prompt = kwargs.get('stdin')
                if prompt:
                    destination = self.out/'operator-controls'/'review-prompts'/(key+'.md')
                    guidance = (
                        '\n\nOperator scheduling guidance: spend about five minutes on evidence '
                        'gathering, then prioritize returning the complete structured proposal. '
                        'Use the cumulative journal and inspect only targeted diagnostic intervals; '
                        'do not repeat a full historical audit at every round. The execution cap '
                        'is at most 30 minutes and may be shorter under the remaining campaign '
                        'budget. Prefer one complete, testable hypothesis to an unfinished large '
                        'proposal. Tool failures are infrastructure failures, not policy evidence. '
                        'Historical blocked reviews did not establish stagnation.\n')
                    contents = Path(prompt).read_text(encoding='utf-8') + guidance
                    if destination.exists() and destination.read_text(encoding='utf-8') != contents:
                        raise ValueError('Recorded operational review prompt changed')
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_text(contents, encoding='utf-8')
                    kwargs['stdin'] = destination
            return super().execute(key, seconds, builder, **kwargs)

    receipt = args.out/'operator-controls'/('runtime-adapter-scaled.json' if args.throughput_controls and controls['max_pods'] > 4 else
        'runtime-adapter-throughput.json' if args.throughput_controls else
        'runtime-adapter-four-pods.json' if args.fleet_controls else 'runtime-adapter.json')
    identity = dict(source=str(args.source.resolve()),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    maximum_review_seconds=1800, fallback='candidate catalog intersection',
                    frozen_evaluator_unchanged=True)
    if args.fleet_controls:
        identity['fleet_controls_sha256'] = hashlib.sha256(args.fleet_controls.read_bytes()).hexdigest()
    if args.throughput_controls:
        identity['throughput_controls_sha256'] = hashlib.sha256(args.throughput_controls.read_bytes()).hexdigest()
        identity['dispatch_sha256'] = hashlib.sha256(Path(__file__).with_name('research_dispatch.py').read_bytes()).hexdigest()
        identity['async_study_sha256'] = hashlib.sha256(Path(__file__).with_name('research_async_study.py').read_bytes()).hexdigest()
    if receipt.exists() and read(receipt) != identity:
        raise ValueError('Operational runtime changed; create a new operator handoff')
    write(receipt, identity)
    runtime.ResilientSupervisor = OperationalSupervisor
    runtime.run(args.out.resolve())


if __name__ == '__main__':
    main()
