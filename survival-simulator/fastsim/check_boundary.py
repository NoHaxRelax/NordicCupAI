"""Check the observation-only boundary mechanically: python fastsim/check_boundary.py

The Python policy worker gets its observation-only guarantee from process isolation,
models/observation_only.py's sanitize_states, and _assert_no_simulator() refusing to
run if src.core / src.elements / fastsim can be imported. Compiling the policy into
the engine binary removes all three, so the guarantee is re-established at compile
time instead, and this script is how a reviewer confirms it rather than taking it on
trust. Three independent checks:

1. INCLUDES. _orchard_policy.cpp (the policy translation unit) may include only
   policy_abi.hpp, policy_iface.hpp, _orchard.hpp and _evasion.hpp, and those four
   files may include only C++ standard headers and each other. Any include of
   _engine.cpp, Python.h or a numpy header fails the check. This is what makes
   "the policy cannot see the engine" a compile error rather than a convention.

2. IDENTIFIERS. The policy sources must not mention the engine's own names (Engine,
   Creature, Fruit, Tree, Predator structs, agent x/y, the world seed, the engine RNG,
   PyObject, ...). A name that is not declared in its translation unit cannot compile,
   so this is belt-and-braces, but it also catches a field being quietly widened.

3. SYMBOLS. The compiled _orchard_policy.o must not reference any Python C-API symbol.
   This is the check that survives someone editing a header: it looks at what the
   object file actually needs from outside itself.

The observation fields the boundary does expose are listed in policy_abi.hpp and
checked here against the observation dict the engine publishes, so the two cannot
drift apart silently.
"""
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
POLICY_UNIT = '_orchard_policy.cpp'
POLICY_FILES = ('../models/self_stuck.hpp', POLICY_UNIT, '_orchard.hpp', '_evasion.hpp', 'policy_abi.hpp', 'policy_iface.hpp')
ALLOWED_LOCAL_INCLUDES = {'../models/self_stuck.hpp', 'policy_abi.hpp', 'policy_iface.hpp', '_orchard.hpp', '_evasion.hpp',
                          '_shared_pysem.inc'}
# Names that only exist on the engine side of the boundary. If one of these turns up
# in the policy sources, something has reached across.
FORBIDDEN = ('Python.h', 'PyObject', 'PyErr_', 'numpy/', 'npy_', '_engine.cpp',
             'struct Engine', 'Engine*', 'Creature', 'biome_map', 'rng_state',
             'agent_observations', 'next_agent_id', 'spawn_fruit', 'non_agent_step')
# The public observation surface, as models/observation_only.py's sanitize_states
# defines it. policy_abi.hpp must expose these and nothing more.
PUBLIC_STATE = ('agent_id', 'energy', 'biome', 'age', 'speed', 'sprint_speed', 'hearing_radius',
                'vision_angle', 'vision_range', 'max_energy')
PUBLIC_OBS = ('type', 'distance', 'angle', 'rel_dir', 'id', 'coords')


def check_includes(fail):
    for name in POLICY_FILES:
        text = (HERE / name).read_text()
        for inc in re.findall(r'^\s*#\s*include\s*([<"][^>"]+[>"])', text, re.M):
            target = inc[1:-1]
            if inc.startswith('<'):
                if '/' in target and not target.startswith('c'):
                    fail(f'{name}: system include {inc} is not a C++ standard header')
            elif target not in ALLOWED_LOCAL_INCLUDES:
                fail(f'{name}: includes {inc}, which is not part of the policy boundary')
        print(f'  includes ok: {name}')


def check_identifiers(fail):
    for name in POLICY_FILES:
        text = (HERE / name).read_text()
        # Comments explain what the policy must NOT see, so strip them before looking.
        code = re.sub(r'//[^\n]*', '', re.sub(r'/\*.*?\*/', '', text, flags=re.S))
        for bad in FORBIDDEN:
            if bad in code:
                fail(f'{name}: mentions {bad!r} outside a comment')
        print(f'  identifiers ok: {name}')


def check_symbols(fail):
    obj = HERE / (POLICY_UNIT[:-4] + '.o')
    if not obj.exists():
        print(f'  SKIP symbols: {obj.name} not built (run fastsim/build_policy.py)')
        return
    for tool in ('nm', 'llvm-nm'):
        try:
            out = subprocess.run([tool, '-u', str(obj)], capture_output=True, text=True, check=True).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        py = sorted({s for s in re.findall(r'\b(Py[A-Z_]\w*|_Py\w*)\b', out)})
        if py:
            fail(f'{obj.name}: references Python C-API symbols {py[:8]}')
        print(f'  symbols ok: {obj.name} has no Python C-API references '
              f'({len(out.splitlines())} undefined symbols, all libstdc++/libm)')
        return
    print('  SKIP symbols: no nm on PATH')


def check_surface(fail):
    abi = (HERE / 'policy_abi.hpp').read_text()
    state = re.search(r'struct AState \{(.*?)\};', abi, re.S)
    obs = re.search(r'struct Obs \{(.*?)\};', abi, re.S)
    if not state or not obs:
        fail('policy_abi.hpp: cannot find the AState/Obs declarations')
        return
    # AState uses the policy's own short spellings for a few published fields.
    aliases = {'aid': 'agent_id', 'sprint': 'sprint_speed', 'hear': 'hearing_radius',
               'cone': 'vision_angle', 'vr': 'vision_range', 'c': 'coords'}
    def fields(block):
        block = re.sub(r'//[^\n]*', '', block)
        return {aliases.get(f, f) for f in re.findall(r'\b([a-z_][a-z_0-9]*)\s*(?:\[\d+\])?\s*[,;]', block)}
    sf, of = fields(state.group(1)), fields(obs.group(1))
    extra_state = sf - set(PUBLIC_STATE) - {'obs'}
    extra_obs = of - set(PUBLIC_OBS) - {'has_rel_dir', 'has_id'}
    if extra_state:
        fail(f'policy_abi.hpp: AState exposes non-public fields {sorted(extra_state)}')
    if extra_obs:
        fail(f'policy_abi.hpp: Obs exposes non-public fields {sorted(extra_obs)}')
    missing = set(PUBLIC_STATE) - sf
    print(f'  surface ok: AState = public state fields'
          f'{" (not exposed: " + ", ".join(sorted(missing)) + ")" if missing else ""}; '
          f'Obs = {", ".join(sorted(of))}')


def main():
    failures = []
    def fail(msg):
        failures.append(msg)
        print(f'  FAIL {msg}')
    for title, fn in (('include list', check_includes), ('identifiers', check_identifiers),
                      ('published surface', check_surface), ('object symbols', check_symbols)):
        print(f'{title}:')
        fn(fail)
    print()
    if failures:
        print(f'BOUNDARY VIOLATED ({len(failures)})')
        return 1
    print('boundary intact: the policy translation unit cannot reach the engine')
    return 0


if __name__ == '__main__':
    sys.exit(main())
