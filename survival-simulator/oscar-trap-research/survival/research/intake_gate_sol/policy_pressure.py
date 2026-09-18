"""Three-arrival gate with a contact-pressure fallback.

The preferred commit remains a complete sensor-derived sleep window.  If the
incoming predator reaches hearing range of its temporary holder before that
window is confirmed, the guide commits immediately rather than being eaten at
the staging point.  Capacity is deliberately capped at three; the wall retires
after that count.
"""
from policy_stationary import IntakeGatePolicy as _Stationary


class IntakeGatePolicy(_Stationary):
    def __init__(self, capacity=3):
        super().__init__(capacity=min(capacity, 3))

    def _update_gate(self, station, states):
        super()._update_gate(station, states)
        if station['seen'] >= self.capacity:
            station['sleep_ready'] = False
            return
        pressure = any(
            aid not in station['holders'] and
            any(o['type'] == 'Predator' and o['distance'] < 50
                for o in state['observations'])
            for aid, state in states.items()
        )
        if pressure and not station['sleep_ready']:
            station['sleep_ready'] = True
            self._event('contact_pressure_commit_window', observed_old=station['seen'])
