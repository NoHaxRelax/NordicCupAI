"""Search space for the no-trapping campaign; simulator mechanics stay fixed.

Two independent policy families, each with its own configuration tree and its
own `mode` in models/experiment_actor.py:

  orchard_evasion  models/orchard_evasion_policy.py - vendored orchard foraging
                   and population play plus the own-sighting evasion layer.
  expert_harvest   models/optimization_policy.py - the shared-map exploration,
                   planning, crowding and territory stack with coordinated
                   harvest enabled.

Mirrors models/experiment_config.py (defaults/inventory/repair/validate) so the
existing optimizer, coverage reports and parameter inventory work unchanged. The
trapping-only blocks (bystander, guide, safety, navigator) are absent because no
bait, lure or trap-site behaviour exists in either family.
"""
from __future__ import annotations

import ast
import copy
import inspect
import math
from pathlib import Path

from models.survival.orchard_population import OrchardPolicy
from models.exploration.expert_policy import ExpertConfig, load_config, DEFAULT_CONFIG_PATH
from models.exploration.global_planner import PlannerConfig, load_planner_config, DEFAULT_CONFIG_PATH as PLANNER_PATH
from models.experiment_config import leaves, get, put, _constraints

FAMILIES = ('orchard_evasion', 'expert_harvest')

# Defaults are the `with_predators_best` evasion settings measured upstream on
# survival-simulator/oscar-overnight-cpp's separate engine port, not here.
EVASION_DEFAULTS = dict(pred_mode=1, pred_r=70., pred_face_r=80., pred_sprint_r=40.,
                        pred_dodge_r=80., pred_dodge_ang=1.4, pred_turn_max=1.0)

EVASION_RANGES = {
    'evasion.pred_mode': (0, 1),
    'evasion.pred_r': (20., 250.),
    'evasion.pred_face_r': (0., 250.),
    'evasion.pred_sprint_r': (0., 150.),
    'evasion.pred_dodge_r': (0., 200.),
    'evasion.pred_dodge_ang': (0., math.pi),
    'evasion.pred_turn_max': (.1, math.pi),
}


def orchard_defaults():
    values = {name: p.default for name, p in inspect.signature(OrchardPolicy.__init__).parameters.items()
              if name not in ('self', 'seed', '_') and not name.startswith('pred_')}
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in values.items()}


def defaults():
    expert = load_config(DEFAULT_CONFIG_PATH).model_dump()
    # The checked-in default keeps harvest off so a trapping colony stays in pure
    # exploration until it finds a site. Without trapping there is no site to wait for.
    expert['harvest']['enabled'] = True
    planner = load_planner_config(PLANNER_PATH).model_dump()
    return dict(orchard=orchard_defaults(), evasion=dict(EVASION_DEFAULTS),
                expert=expert, planner=planner)


def orchard_kwargs(config):
    """Constructor keywords for OrchardEvasionPolicy from the orchard+evasion blocks."""
    values = {key: math.inf if value is None else value for key, value in config['orchard'].items()}
    values.update(config['evasion'])
    return values


def _read_parameters():
    """Orchard constructor keys actually read through self.P, via the source AST."""
    tree = ast.parse(Path(inspect.getfile(OrchardPolicy)).read_text(encoding='utf-8'))
    return {n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)
            and ((isinstance(n.value, ast.Attribute) and n.value.attr == 'P')
                 or (isinstance(n.value, ast.Name) and n.value.id == 'P'))}


def inventory():
    """Catalog every exposed scalar, including fixed/dormant values with reasons."""
    baseline = defaults()
    used = _read_parameters()
    constraints = _constraints(ExpertConfig.model_validate(baseline['expert']), 'expert')
    constraints.update(_constraints(PlannerConfig.model_validate(baseline['planner']), 'planner'))
    result = []
    for path, value in leaves(baseline):
        block, _, leaf = path.partition('.')
        reason = None
        if path.startswith(('expert.mechanics.', 'planner.estimator.biome_movement_factors.')):
            reason = 'Known simulator mechanics, not policy hyperparameters'
        elif path in ('planner.mapping_enabled', 'planner.draw_overlay',
                      'planner.estimator.boundary_wall_thickness'):
            reason = 'Coordinator invariant, rendering option, or known wall geometry'
        elif path == 'expert.harvest.enabled':
            reason = 'Coordinated harvest is the point of this family; switching it off is a separate control'
        elif block == 'orchard' and leaf not in used:
            reason = 'Stored by the Orchard constructor but never read'
        if isinstance(value, str):
            result.append(dict(path=path, default=value, kind='str', low=value, high=value,
                               tunable=False, reason=reason or 'Categorical setting; this search space is numeric',
                               family='orchard_evasion', block=block, nullable=False))
            continue
        kind = 'bool' if isinstance(value, bool) else 'int' if isinstance(value, int) else 'float'
        numeric = 1. if value is None else float(value)
        low, high = (.5*numeric, 2*numeric) if numeric > 0 else (0., 1.)
        if kind == 'int':
            low, high = max(1, math.floor(low)), max(2, math.ceil(high))
        if path.endswith(('.no_eat_age', '.heir_age')):
            low, high = 40., 120.
        elif path.endswith('.late_still_t'):
            low, high = 600., 3000.
        elif path.endswith('.post_radius'):
            low, high = 12., 65.
        elif path.endswith('.cap_min'):
            low, high = 1, 8
        elif path.endswith('.cap_max'):
            low, high = 8, 50
        elif path.endswith('.reserve_t1'):
            low, high = 900., 3000.
        elif path.endswith('.dump_after_t'):
            low, high = 300., 3000.
        if path in EVASION_RANGES:
            low, high = EVASION_RANGES[path]
        bounds = constraints.get(path, {})
        low = max(low, bounds.get('ge', -math.inf), bounds.get('gt', -math.inf) + (1 if kind == 'int' else 1e-6))
        high = min(high, bounds.get('le', math.inf), bounds.get('lt', math.inf) - (1 if kind == 'int' else 1e-6))
        if kind == 'int':
            low, high = math.ceil(low), math.floor(high)
        result.append(dict(path=path, default=value, kind=kind, low=low, high=high,
                           tunable=reason is None, reason=reason,
                           family='expert_harvest' if block in ('expert', 'planner') else 'orchard_evasion',
                           block='.'.join(path.split('.')[:2]) if block in ('expert', 'planner') else block,
                           nullable=value is None))
    return result


def repair(config):
    """Coupled bounds; never changes simulator parameters or physical constants."""
    config = copy.deepcopy(config)
    p = config['orchard']
    p['cap_max'] = max(p['cap_min'], p['cap_max'])
    p['reserve_t1'] = max(p['reserve_t0'] + 1., p['reserve_t1'])
    b = config['planner']['biome_inference']
    b['max_refit_interval_seconds'] = max(b['refit_interval_seconds'], b['max_refit_interval_seconds'])
    b['max_samples'] = max(b['max_samples'], b['min_samples'])
    return config


def validate(config):
    baseline = defaults()
    expected, supplied = dict(leaves(baseline)), dict(leaves(config))
    if set(expected) != set(supplied):
        raise ValueError(f'Configuration fields differ: {set(expected) ^ set(supplied)}')
    for path, value in supplied.items():
        original = expected[path]
        if value is None and original is None:
            continue
        if isinstance(original, str):
            if value != original:
                raise ValueError(f'{path} is a fixed categorical setting')
        elif isinstance(original, bool):
            if not isinstance(value, bool):
                raise ValueError(f'{path} must be boolean')
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{path} must be finite and nonnegative')
        elif isinstance(original, int) and not isinstance(value, int):
            raise ValueError(f'{path} must be integral')
    for spec in inventory():
        if not spec['tunable'] and supplied[spec['path']] != expected[spec['path']]:
            raise ValueError(f"Fixed field changed: {spec['path']}")
    ExpertConfig.model_validate(config['expert'])
    PlannerConfig.model_validate(config['planner'])
    if repair(config) != config:
        raise ValueError('Inconsistent coupled parameter bounds')
    for name in ('tree_half', 'tree_slots', 'cap_min', 'cap_max'):
        if config['orchard'][name] <= 0:
            raise ValueError(f'orchard.{name} must be positive')
    return config
