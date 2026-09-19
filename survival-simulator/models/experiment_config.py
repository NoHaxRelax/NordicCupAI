"""Experiment parameters; simulator mechanics are deliberately not search variables."""
from __future__ import annotations

import ast
import copy
import inspect
import math
from pathlib import Path

from models.survival.oscar_orchard import OrchardPolicy
from models.exploration.expert_policy import ExpertConfig, load_config, DEFAULT_CONFIG_PATH
from models.exploration.global_planner import PlannerConfig, load_planner_config, DEFAULT_CONFIG_PATH as PLANNER_PATH
from models.entrapment.bystander_avoidance import DEFAULT_CONFIG as BYSTANDER_DEFAULTS

FEATURES = (
    'consistent_escape', 'escape_memory', 'shared_danger', 'trap_exclusion',
    'safe_steering', 'risk_aware_food', 'role_budget', 'refresh_guide_geometry',
    'emergency_reproduction', 'late_conservation',
    'corner_pockets',
)


def defaults():
    orchard = {name: p.default for name, p in inspect.signature(OrchardPolicy.__init__).parameters.items()
               if name not in ('self', 'seed', '_')}
    orchard = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in orchard.items()}
    trapping = dict(orchard, extra_old=False, heir_age=1e9)
    expert = load_config(DEFAULT_CONFIG_PATH).model_dump()
    expert['harvest']['enabled'] = False
    planner = load_planner_config(PLANNER_PATH).model_dump()
    planner['population_after_alignment'] = False
    return dict(
        features=dict.fromkeys(FEATURES, False), trapping_orchard=trapping,
        expert=expert, planner=planner, bystander=dict(BYSTANDER_DEFAULTS),
        navigator=dict(cell_size=16., max_cells=20000, clearance=5.05, stall_seconds=5.),
        guide=dict(hearing_target=55., vision_target=235., half_cone_target=math.radians(25),
                   safe_distance=48., bait_buffer=5., contact_buffer=18., trapped_radius=40.,
                   lost_wait_ticks=5, reacquire_arrival_distance=12., delivery_arrival_distance=1.,
                   predator_clearance=11., position_tolerance=.75, heading_tolerance=.05,
                   mismatch_ticks=2),
        safety=dict(danger_radius=130., memory_seconds=2., shared_ttl=4.,
                    shared_radius=100., uncertainty_limit=8., uncertainty_padding=1.,
                    trap_radius=110., lure_radius=65., angle_candidates=24,
                    wall_margin=6., lookahead_seconds=.5, sprint_reserve=100.,
                    guide_fraction=.25, minimum_workers=3, guide_timeout=35.,
                    guide_lost_seconds=5., guide_cooldown=8., guide_geometry_interval=2.,
                    young_age=40., emergency_target_fraction=.5, emergency_min_young=2,
                    emergency_parent_max_age=85., emergency_reserve=30.,
                    emergency_births_per_tick=1, conservation_start=600.,
                    conservation_end=2400., idle_turn_fraction=.2),
    )


def orchard_kwargs(values):
    return {key: math.inf if value is None else value for key, value in values.items()}


def leaves(value, prefix=''):
    for key, child in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(child, dict):
            yield from leaves(child, path)
        else:
            yield path, child


def get(config, path):
    for part in path.split('.'):
        config = config[part]
    return config


def put(config, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        config = config[part]
    config[parts[-1]] = value


def _constraints(model, prefix):
    result = {}
    for key, value in model.__dict__.items():
        path = f'{prefix}.{key}'
        if hasattr(value, 'model_fields'):
            result.update(_constraints(value, path))
        else:
            bounds = {}
            for metadata in type(model).model_fields[key].metadata:
                for name in ('ge', 'gt', 'le', 'lt'):
                    bound = getattr(metadata, name, None)
                    if bound is not None:
                        bounds[name] = bound
            result[path] = bounds
    return result


def inventory():
    """Catalog every exposed scalar, including fixed/dormant values with reasons.

    Constructor parameters that are stored but never read by Orchard are not
    given fake optimization dimensions. Pydantic defaults not in JSON are included.
    """
    baseline = defaults()
    tree = ast.parse(Path(inspect.getfile(OrchardPolicy)).read_text(encoding='utf-8'))
    used = {n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)
            and ((isinstance(n.value, ast.Attribute) and n.value.attr == 'P')
                 or (isinstance(n.value, ast.Name) and n.value.id == 'P'))}
    constraints = _constraints(ExpertConfig.model_validate(baseline['expert']), 'expert')
    constraints.update(_constraints(PlannerConfig.model_validate(baseline['planner']), 'planner'))
    # Physically meaningful ranges for new knobs and exceptional/sentinel defaults.
    ranges = {
        'bystander.angle_candidates': (12, 64),
        'bystander.contact_clearance': (30.1, 60.),
        'bystander.hearing_clearance': (60., 140.),
        'bystander.vision_clearance': (250., 350.),
        'bystander.vision_half_angle': (math.pi/6, math.pi/2),
        'bystander.trap_clearance': (60., 180.),
        'bystander.trap_check_distance': (90., 240.),
        'safety.danger_radius': (70., 220.), 'safety.memory_seconds': (.5, 6.),
        'safety.shared_ttl': (.5, 12.), 'safety.shared_radius': (50., 180.),
        'safety.trap_radius': (50., 200.), 'safety.lure_radius': (30., 130.),
        'safety.angle_candidates': (8, 48), 'safety.guide_fraction': (.05, .5),
        'safety.minimum_workers': (1, 8), 'safety.wall_margin': (5.05, 12.),
        'safety.guide_timeout': (10., 120.), 'safety.lookahead_seconds': (.1, 1.5),
        'safety.emergency_target_fraction': (.1, 1.), 'safety.emergency_min_young': (1, 6),
        'safety.emergency_parent_max_age': (45., 110.), 'safety.idle_turn_fraction': (0., 1.),
        'guide.hearing_target': (35., 59.), 'guide.vision_target': (150., 249.),
        'guide.half_cone_target': (.2, math.pi/6), 'guide.contact_buffer': (15.1, 30.),
        'guide.safe_distance': (30., 90.), 'guide.predator_clearance': (10.05, 16.),
        'navigator.clearance': (5.05, 12.),
    }
    result = []
    for path, value in leaves(baseline):
        reason = None
        if path.startswith(('expert.mechanics.', 'planner.estimator.biome_movement_factors.')):
            reason = 'Known simulator mechanics, not policy hyperparameters'
        elif path.startswith('expert.harvest.'):
            reason = 'Disabled by the trapping coordinator; Orchard owns harvesting'
        elif path in ('planner.population_after_alignment', 'planner.mapping_enabled', 'planner.draw_overlay',
                      'planner.estimator.boundary_wall_thickness'):
            reason = 'Coordinator invariant, rendering option, or known wall geometry'
        elif path.split('.')[0] == 'trapping_orchard' and path.split('.')[1] not in used:
            reason = 'Stored by the current Orchard constructor but never read'
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
        if path in ranges:
            low, high = ranges[path]
        bounds = constraints.get(path, {})
        low = max(low, bounds.get('ge', -math.inf), bounds.get('gt', -math.inf) + (1 if kind == 'int' else 1e-6))
        high = min(high, bounds.get('le', math.inf), bounds.get('lt', math.inf) - (1 if kind == 'int' else 1e-6))
        if kind == 'int':
            low, high = math.ceil(low), math.floor(high)
        result.append(dict(path=path, default=value, kind=kind, low=low, high=high,
                           tunable=reason is None, reason=reason,
                           block='.'.join(path.split('.')[:2]) if path.startswith(('expert.', 'planner.')) else path.split('.')[0],
                           nullable=value is None))
    return result


def repair(config):
    """Coupled bounds; never changes simulator parameters or physical constants."""
    config = copy.deepcopy(config)
    for name in ('trapping_orchard',):
        p = config[name]
        p['cap_max'] = max(p['cap_min'], p['cap_max'])
        p['reserve_t1'] = max(p['reserve_t0'] + 1., p['reserve_t1'])
    b = config['planner']['biome_inference']
    b['max_refit_interval_seconds'] = max(b['refit_interval_seconds'], b['max_refit_interval_seconds'])
    b['max_samples'] = max(b['max_samples'], b['min_samples'])
    s = config['safety']
    s['conservation_end'] = max(s['conservation_start'] + 1., s['conservation_end'])
    config['guide']['safe_distance'] = max(config['guide']['contact_buffer'], config['guide']['safe_distance'])
    b = config['bystander']
    b['trap_check_distance'] = max(b['trap_check_distance'], b['trap_clearance']+20.)
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
        if isinstance(original, bool):
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
    for key in ('trapping_orchard',):
        for name in ('tree_half', 'tree_slots', 'cap_min', 'cap_max'):
            if config[key][name] <= 0:
                raise ValueError(f'{key}.{name} must be positive')
    if (config['safety']['angle_candidates'] < 4 or config['bystander']['angle_candidates'] < 4
            or config['navigator']['cell_size'] <= 0):
        raise ValueError('Invalid steering/grid resolution')
    return config
