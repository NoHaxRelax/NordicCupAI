"""Compile the native engine+policy module: python fastsim/build_policy.py

Same flags as build.py (they are what make the engine bit-identical to numpy), but
builds _policy.cpp + _orchard_policy.cpp to _policy<EXT_SUFFIX> and records
build-info-policy.json. It never writes _engine.cpp, _engine<EXT_SUFFIX> or
build-info.json, so the engine's recorded provenance keeps matching the binary the
existing benchmarks were measured on.

Two translation units on purpose:
  _policy.cpp          sees the engine (it #includes _engine.cpp) and the Python C API
  _orchard_policy.cpp  sees ONLY policy_abi.hpp - the observation-only boundary
They meet at the polabi::IPolicy vtable declared in policy_iface.hpp.

The build also regenerates _shared_pysem.inc (the CPython-semantics helpers the policy
needs: MT19937, set iteration order, float % and //, hash) out of _engine.cpp and fails
if the slice has drifted, so the two copies cannot diverge unnoticed.
"""
import os, re, subprocess, sys, sysconfig, pathlib, hashlib, json, shlex
import numpy

HERE = pathlib.Path(__file__).resolve().parent
UNITS = ('_policy.cpp', '_orchard_policy.cpp')
HEADERS = ('_engine.cpp', '_orchard.hpp', '_evasion.hpp', 'policy_abi.hpp', 'policy_iface.hpp',
           '_shared_pysem.inc', '../models/self_stuck.hpp')
EXTRA_FLAGS = shlex.split(os.environ.get('FASTSIM_EXTRA_FLAGS', ''))
FLAGS = ['-O2', '-std=c++17', '-fPIC', '-ffp-contract=off', '-fno-fast-math',
         '-fno-builtin-sin', '-fno-builtin-cos', '-fno-builtin-sincos', *EXTRA_FLAGS]


def refresh_shared(check=True):
    """Re-slice the CPython-semantics helpers out of _engine.cpp into _shared_pysem.inc."""
    src = (HERE / '_engine.cpp').read_text().splitlines()
    try:
        i = next(n for n, l in enumerate(src) if l.startswith('const double PI = 3.141592653589793;'))
        j = next(n for n, l in enumerate(src) if 'GEOS robust orientation' in l)
    except StopIteration:
        raise SystemExit('build_policy: cannot find the shared-helper slice in _engine.cpp; '
                         'update refresh_shared() before building')
    while not src[j].startswith('// ---'):
        j -= 1
    # The slice is included by two translation units, so its handful of non-inline free
    # functions (py_hash_double and friends) have to become inline. A definition that
    # starts a line and is not already inline/static/struct/etc is one of them.
    DEF = re.compile(r'^(?!inline|static|constexpr|struct|class|enum|namespace|template)'
                     r'[A-Za-z_][\w:*&<> ]*\s\*?\w+\([^;]*\)\s*\{')
    lines = ['inline ' + l if DEF.match(l) else l for l in src[i:j]]
    body = '\n'.join(lines).rstrip() + '\n'
    out = HERE / '_shared_pysem.inc'
    if check and out.exists() and out.read_text() != body:
        raise SystemExit(f'build_policy: {out.name} no longer matches _engine.cpp lines {i+1}-{j}. '
                         'Re-run with FASTSIM_REFRESH_SHARED=1 once you have checked the diff.')
    out.write_text(body, newline='\n')  # LF, like every other source here


def build(verbose=True):
    refresh_shared(check=not os.environ.get('FASTSIM_REFRESH_SHARED'))
    out = HERE / ('_policy' + sysconfig.get_config_var('EXT_SUFFIX'))
    cxx = os.environ.get('CXX', 'g++' if sys.platform == 'win32' else 'c++')
    flags = FLAGS + ['-I', sysconfig.get_paths()['include'], '-I', numpy.get_include()]
    env = dict(os.environ)
    if sys.platform == 'darwin':
        link_flags = ['-bundle', '-undefined', 'dynamic_lookup']
        env.setdefault('DEVELOPER_DIR', '/Library/Developer/CommandLineTools')
    else:
        link_flags = ['-shared']
    link = []
    if sys.platform == 'win32':
        link = ['-L', str(pathlib.Path(sys.base_prefix) / 'libs'),
                '-lpython' + str(sys.version_info.major) + str(sys.version_info.minor), '-static']
    objs, cmds = [], []
    for unit in UNITS:
        obj = HERE / (unit[:-4] + '.o')
        # The policy unit gets no Python/numpy include path at all: another way the
        # boundary is enforced rather than promised.
        unit_flags = flags if unit == '_policy.cpp' else FLAGS
        cmd = [cxx, *unit_flags, '-c', str(HERE / unit), '-o', str(obj)]
        cmds.append(cmd)
        if verbose:
            print(' '.join(cmd))
        subprocess.run(cmd, check=True, env=env)
        objs.append(str(obj))
    tmp = out.with_suffix('.tmp')
    cmd = [cxx, *link_flags, *EXTRA_FLAGS, *objs, *link, '-o', str(tmp)]
    cmds.append(cmd)
    if verbose:
        print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)
    os.replace(tmp, out)  # new inode: running processes keep their loaded copy
    (HERE / 'build-info-policy.json').write_text(json.dumps(dict(
        sources={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in UNITS + HEADERS},
        binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        python=sys.version, numpy=numpy.__version__, commands=cmds,
        compiler=subprocess.check_output([cxx, '--version'], text=True).splitlines()[0]), indent=2) + '\n')
    return out


if __name__ == '__main__':
    print(build())
