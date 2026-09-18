"""Sparse-arrival variant: hold the incoming predator on the wall normal.

The base controller supplies all observation-only mapping and gate logic.  At
the temporary holder, this variant stays 90 units from the near face instead
of orbiting.  That preserves the incoming predator's wall-normal approach so
it can follow the guide's short final sprint when the old mass sleeps.
"""
from policy import IntakeGatePolicy as _Base
from controller import add, mul


class IntakeGatePolicy(_Base):
    def _temporary_target(self, pose, state, face, outward, worker):
        threats = [o for o in state['observations'] if o['type'] == 'Predator']
        return add(face, mul(outward, 90)), threats
