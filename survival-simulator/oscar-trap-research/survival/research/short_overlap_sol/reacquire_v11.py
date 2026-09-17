"""Frozen v10 plus bounded native-observation reacquisition after DTO loss."""
import hashlib
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "simple_chase" / "policy_v10_approach_lane.py"
EXPECTED = "7e6cfad20eea082aa83e66a8cb505fe9ee9d19a5efb6aa601d6877a8bbd2d1e3"
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
    raise RuntimeError("frozen v10 dependency changed")

source = SOURCE.read_text()
needle = '''            # Keep predator in hearing range across rest. DTO is one movement stale.
            if distance>75:
'''
replacement = '''            # If the ordinary Predator DTO has been absent for a full second,
            # stale distance must not authorize indefinite holding. Walk toward
            # the last observed point, then sweep a small free ring around it,
            # continuously scanning. A fresh DTO immediately returns control to
            # the unchanged v10 latency-safe spacing logic.
            lost_timeout = nearest is None and sim_time-self.last_predator_seen>1.
            if lost_timeout:
                if math.dist(pose.p,pred)>12:
                    lost_path=self._path(pose.p,pred,5.01)
                    lost_target=lost_path[0] if lost_path else pose.p
                else:
                    lost_target=pose.p
                    for offset in range(16):
                        angle=math.tau*((self.act_tick//4+offset)%16)/16
                        candidate=add(pred,(30*math.cos(angle),30*math.sin(angle)))
                        if self._free(candidate,5.01) and self._clear(pose.p,candidate,5.01):
                            lost_target=candidate;break
                scan_turn=wrap(turn+(-1.,0.,1.,0.)[self.act_tick%4]*math.pi/3)
                actions.append(self._move(aid,state,pose,lost_target,
                    'bounded_last_observation_reacquisition',turn=scan_turn,sprint=False))
                continue
            # Keep predator in hearing range across rest. DTO is one movement stale.
            if distance>75:
'''
if source.count(needle) != 1:
    raise RuntimeError("v10 injection point changed")
source = source.replace(needle, replacement)
namespace = {"__name__": __name__, "__file__": __file__}
exec(compile(source, str(SOURCE), "exec"), namespace)
SimpleChase = namespace["SimpleChase"]

