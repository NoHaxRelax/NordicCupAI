"""Run Orchard behind a JSON-only, spawned-process observation boundary.

The worker gets public agent observations, elapsed simulation time, and a
separate policy RNG seed. It never receives a simulator, world seed, engine
objects, or unobserved world data. This is an accidental-information-leak
boundary, not an operating-system sandbox against malicious Python code.
"""
from __future__ import annotations

import inspect
import json
import math
import multiprocessing
from numbers import Integral, Real
import os
import sys


AGENT_FIELDS = (
    "agent_id", "energy", "biome", "age", "speed", "sprint_speed",
    "hearing_radius", "vision_angle", "vision_range", "max_energy", "observations",
)
OBSERVATION_FIELDS = {
    "Fruit": ("type", "distance", "angle"),
    "Tree": ("type", "distance", "angle"),
    "Agent": ("type", "distance", "angle", "id", "rel_dir"),
    "Predator": ("type", "distance", "angle", "rel_dir"),
    "Edge": ("type", "coords"),
}
ACTION_FIELDS = (
    "agent_id", "move_distance", "move_direction", "turn_angle", "spawn_agent",
)
TIMEOUT_SECONDS = 120.0


def _number(value, field, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return value


def _identifier(value, field):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def _required(mapping, field):
    try:
        return mapping[field]
    except KeyError as exc:
        raise ValueError(f"Missing public observation field: {field}") from exc


def _observation(item):
    if not isinstance(item, dict):
        raise ValueError("Each observation must be a dictionary")
    kind = _required(item, "type")
    if not isinstance(kind, str) or kind not in OBSERVATION_FIELDS:
        raise ValueError(f"Unknown public observation type: {kind!r}")
    clean = {"type": kind}
    if kind == "Edge":
        coords = _required(item, "coords")
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ValueError("Edge.coords must contain two (x, y) endpoints")
        clean["coords"] = []
        for point in coords:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Edge.coords must contain two (x, y) endpoints")
            clean["coords"].append([_number(v, "Edge.coords") for v in point])
    else:
        clean["distance"] = _number(_required(item, "distance"), "distance", minimum=0)
        clean["angle"] = _number(_required(item, "angle"), "angle")
        if kind == "Agent":
            clean["id"] = _identifier(_required(item, "id"), "id")
        if kind in ("Agent", "Predator"):
            clean["rel_dir"] = _number(_required(item, "rel_dir"), "rel_dir")
    return clean


def sanitize_states(states_list):
    """Copy only ObservationResponse fields and official nested sensing fields.

    Unknown extra fields are discarded without accessing their values. Unknown
    object types and malformed or missing required public fields are rejected.
    """
    if not isinstance(states_list, (list, tuple)):
        raise ValueError("states_list must be a list of observation dictionaries")
    clean_states = []
    seen_ids = set()
    for state in states_list:
        if not isinstance(state, dict):
            raise ValueError("Each agent state must be a dictionary")
        clean = {"agent_id": _identifier(_required(state, "agent_id"), "agent_id")}
        if clean["agent_id"] in seen_ids:
            raise ValueError("Duplicate agent_id in observations")
        seen_ids.add(clean["agent_id"])
        biome = _required(state, "biome")
        if not isinstance(biome, str) or not biome:
            raise ValueError("biome must be a nonempty string")
        for key in AGENT_FIELDS:
            if key in ("agent_id", "biome", "observations"):
                continue
            clean[key] = _number(_required(state, key), key,
                                 minimum=None if key == "energy" else 0)
        if clean["max_energy"] == 0:
            raise ValueError("max_energy must be positive")
        clean["biome"] = biome
        observations = _required(state, "observations")
        if not isinstance(observations, (list, tuple)):
            raise ValueError("observations must be a list")
        clean["observations"] = [_observation(item) for item in observations]
        clean_states.append(clean)
    return clean_states


def _encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _decode(value):
    def reject_constant(constant):
        raise ValueError(f"Non-finite JSON constant: {constant}")
    return json.loads(value.decode("utf-8"), parse_constant=reject_constant)


def _assert_no_simulator():
    forbidden = [name for name in sys.modules
                 if name == "src.core" or name.startswith("src.core.")
                 or name == "src.elements" or name.startswith("src.elements.")
                 or name == "fastsim" or name.startswith("fastsim.")]
    if forbidden:
        raise RuntimeError(f"Simulator modules loaded in policy worker: {forbidden}")


def _worker(connection):
    try:
        _assert_no_simulator()
        if not connection.poll(TIMEOUT_SECONDS):
            raise TimeoutError("No policy startup message received")
        startup = _decode(connection.recv_bytes())
        if startup.get("kind") != "start":
            raise ValueError("Expected policy startup message")
        policy_seed = _identifier(startup["policy_seed"], "policy_seed")
        kwargs = startup["policy_kwargs"]
        sys.argv = ["orchard-policy-worker"]
        from models.survival.oscar_orchard import OrchardPolicy
        from models.observed_bounds import ObservedBoundsOrchard
        from src.utils.DTOs import ObservationResponse
        _assert_no_simulator()
        if set(ObservationResponse.model_fields) != set(AGENT_FIELDS):
            raise RuntimeError("Observation whitelist differs from official ObservationResponse")
        tunables = set(inspect.signature(OrchardPolicy.__init__).parameters) - {"self", "seed", "_"}
        if set(kwargs) - tunables:
            raise ValueError(f"Unknown policy parameters: {sorted(set(kwargs) - tunables)}")
        policy = ObservedBoundsOrchard(seed=policy_seed, **kwargs)
        _assert_no_simulator()
        audit = {
            "worker_pid": os.getpid(),
            "multiprocessing_start_method": multiprocessing.get_start_method(),
            "transport": "json", "policy_seed": policy_seed,
            "agent_fields": list(AGENT_FIELDS),
            "observation_fields": {key: list(fields) for key, fields in OBSERVATION_FIELDS.items()},
            "env_loaded": False, "dimensions_source": "observed_edges",
        }
        connection.send_bytes(_encode({"kind": "ready", "audit": audit}))
        while True:
            request = _decode(connection.recv_bytes())
            if request.get("kind") == "close":
                break
            if request.get("kind") != "step":
                raise ValueError("Unknown policy request")
            states = sanitize_states(request["states"])
            sim_time = _number(request["sim_time"], "sim_time", minimum=0)
            _assert_no_simulator()
            actions = policy(states, sim_time)
            _assert_no_simulator()
            encoded_actions = []
            for aid, action in actions:
                data = {key: getattr(action, key) for key in ACTION_FIELDS}
                if aid != data["agent_id"]:
                    raise ValueError("Policy action identifier mismatch")
                encoded_actions.append([aid, data])
            connection.send_bytes(_encode({
                "kind": "actions", "actions": encoded_actions,
                "audit": {"env_loaded": False, "bounds": policy.bounds_audit},
            }))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send_bytes(_encode({"kind": "error", "error": f"{type(exc).__name__}: {exc}"}))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ObservationOnlyOrchard:
    """Callable observation-only Orchard actor with explicit process cleanup."""

    def __init__(self, policy_seed=0, policy_kwargs=None):
        policy_seed = _identifier(policy_seed, "policy_seed")
        if policy_kwargs is not None and not isinstance(policy_kwargs, dict):
            raise ValueError("policy_kwargs must be a JSON-compatible dictionary")
        startup = _encode({"kind": "start", "policy_seed": policy_seed,
                           "policy_kwargs": policy_kwargs or {}})
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(target=_worker, args=(child,), daemon=True)
        self._closed = False
        self.audit = {}
        self.last_audit = {}
        try:
            self._process.start()
            child.close()
            self._connection.send_bytes(startup)
            response = self._receive("ready")
            self.audit = response["audit"]
            self.last_audit = dict(self.audit)
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, expected):
        try:
            if not self._connection.poll(TIMEOUT_SECONDS):
                raise TimeoutError(f"Policy worker did not reply within {TIMEOUT_SECONDS:g}s")
            response = _decode(self._connection.recv_bytes())
            if response.get("kind") == "error":
                raise RuntimeError(f"Observation-only policy failed: {response['error']}")
            if response.get("kind") != expected:
                raise RuntimeError(f"Unexpected policy response: {response.get('kind')}")
            return response
        except BaseException:
            self.close()
            raise

    def __call__(self, states_list, sim_time):
        if self._closed:
            raise RuntimeError("ObservationOnlyOrchard is closed")
        request = _encode({"kind": "step", "states": sanitize_states(states_list),
                           "sim_time": _number(sim_time, "sim_time", minimum=0)})
        try:
            self._connection.send_bytes(request)
            response = self._receive("actions")
        except BaseException:
            self.close()
            raise
        self.last_audit = {**self.audit, **response["audit"]}
        return [(aid, action) for aid, action in response["actions"]]

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                self._connection.send_bytes(_encode({"kind": "close"}))
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self._connection.close()
            if self._process.pid is not None:
                self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
