"""Check whether a predator sighting supports being its detectable prey.

This is public sensing geometry, not access to the predator's target ID. The
sighting is delayed by a movement, so the next guide tick must still verify
following and protect against capture.
"""
import math
from shapely.geometry import LineString


def detectable_observer(observation, state, edges):
    distance = observation['distance']
    if distance <= 60.: return True
    heading_error = observation.get('rel_dir')
    if heading_error is None or distance > 250.: return False
    if abs(math.atan2(math.sin(heading_error),math.cos(heading_error))) > math.pi/6:
        return False
    # Outside our hearing radius the sighting itself establishes clear LOS.
    if distance > state['hearing_radius']: return True
    point = (distance*math.cos(observation['angle']),distance*math.sin(observation['angle']))
    sight = LineString([(0.,0.),point])
    return not any(sight.intersects(LineString(edge)) for edge in edges)
