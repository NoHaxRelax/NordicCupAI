"""Trusted engine selection and build provenance; never imported by policy workers."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sysconfig
import sys

ROOT = Path(__file__).resolve().parents[1]


def identity(engine, root=ROOT):
    if engine == 'python':
        return dict(name='python', ordering='object-address')
    if engine != 'fastsim':
        raise ValueError(f'Unknown simulation engine: {engine}')
    folder = Path(root)/'fastsim'
    try:
        info = json.loads((folder/'build-info.json').read_text())
        source_hash = hashlib.sha256((folder/'_engine.cpp').read_bytes()).hexdigest()
        binary_hash = hashlib.sha256((folder/('_engine'+sysconfig.get_config_var('EXT_SUFFIX'))).read_bytes()).hexdigest()
    except (OSError, KeyError) as exc:
        raise RuntimeError('Build fastsim on this host first: python fastsim/build.py') from exc
    if info['source_sha256'] != source_hash or info['binary_sha256'] != binary_hash:
        raise RuntimeError('Stale or changed fastsim build; rebuild before preparing a campaign')
    if info['numpy'] != importlib.metadata.version('numpy') or info['python'] != sys.version:
        raise RuntimeError('Python/NumPy differ from the native build; rebuild on this environment')
    return dict(name='fastsim', ordering='creation-counter', source_sha256=source_hash,
                binary_sha256=binary_hash, compiler=info['compiler'], numpy=info['numpy'])


def create(engine, **kwargs):
    identity(engine)
    if engine == 'python':
        from src.core import SimulationCore
    else:
        from fastsim import SimulationCore
    return SimulationCore(**kwargs)


def observations(sim, engine):
    if engine == 'fastsim':
        return sim.state()['observations']
    return [sim.env.get_agent_state(a.agent_id) for a in sim.env.agents]
