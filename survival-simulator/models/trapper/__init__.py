"""Predator trapping on top of a normal foraging society.

Layers (bottom to top):

* ``geometry``        - vector and rectangle helpers.
* ``world``           - the WorldState views every higher layer consumes.
* ``oracle``          - WorldState built from the true engine state (development).
* ``estimator``       - WorldState reconstructed from ordinary observations only.
* ``predator_model``  - exact clone of the engine's predator decision and movement,
                        used to predict where a predator will go.
* ``sites``           - wall and narrow-gap trap site detection from rectangles.
* ``paths``           - grid path planning around obstacles.
* ``lure``            - guide behaviour: attract a predator, lead it, deliver it.
* ``manager``         - trap staffing: holders, successors, guards, deliveries.
* ``policy``          - TrapperPolicy: society baseline + trap manager overrides.
"""
