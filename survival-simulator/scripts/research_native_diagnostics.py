"""Translate the C++ evaluator's measured events to the shared diagnostic schema."""
from types import SimpleNamespace


def consume(collector):
    for kind, aid, other, a, b, c in collector.env.pop_diagnostics():
        agent = SimpleNamespace(agent_id=aid, energy=a)
        if kind == 4:
            collector.register(agent, other)
        elif kind in (0, 1, 2, 3):
            name = ('movement_energy', 'turning_energy', 'reproduction_energy', 'maintenance_energy')[kind]
            collector.ledger(agent, name, a)
            if kind == 0:
                collector.ledger(agent, 'requested_distance', b)
                collector.ledger(agent, 'actual_distance', c)
                if b > .1 and c < .01:
                    collector.ledger(agent, 'blocked_seconds', collector.dt)
        elif kind == 5:
            for name, value in (('fruits_eaten', 1), ('fruit_gross_energy', a),
                                ('fruit_absorbed_energy', b), ('fruit_cap_waste', a-b)):
                collector.ledger(agent, name, value)
            row = collector.agents[aid]
            start = row['last_meal'] if row['last_meal'] is not None else row['born']
            row['longest_meal_gap'] = max(row['longest_meal_gap'], collector.now-start)
            row['last_meal'] = collector.now
            collector.event('fruit_consumed', agent_id=aid, fruit_id=other,
                role=collector.roles.get(str(aid), 'unknown'), gross=a, absorbed=b, cap_waste=a-b,
                age_seconds=collector.now-collector.fruit_birth.pop(other, collector.now))
        elif kind == 6:
            collector.fruit_birth[aid] = collector.now
            collector.totals['fruits_spawned'] += 1
        elif kind == 7:
            collector.fruit_birth.pop(aid, None)
            collector.totals['fruits_rotted'] += 1
            collector.event('fruit_rotted', fruit_id=aid, energy=a)
        elif kind in (8, 9):
            cause = 'energy_depletion' if kind == 8 else 'predation'
            collector.death_causes[cause] += 1
            death = dict(cause=cause, energy=a, age=b, role=collector.roles.get(str(aid), 'unknown'))
            collector.agents[aid]['death'] = dict(sim_time=collector.now, **death)
            collector.event('death', agent_id=aid, predator_id=None if other < 0 else other, **death)
        else:
            raise RuntimeError(f'Unknown native diagnostic event: {kind}')
