"""Compile in place: python survival-simulator/scripts/predator_stuck_cpp/build.py

Needs only a C++17 compiler and the Python headers (no pybind11/setuptools).
-ffp-contract=off keeps every multiply and add separately rounded, like numpy.
The sin/cos builtins are disabled so the compiler cannot merge them into a
sincos call, which may round differently from numpy's separate libm calls.
"""
import os, subprocess, sys, sysconfig, pathlib, hashlib, json
import numpy

HERE = pathlib.Path(__file__).resolve().parent


def build(verbose=True):
    src = HERE / '_stuck.cpp'
    out = HERE / ('_stuck' + sysconfig.get_config_var('EXT_SUFFIX'))
    cxx = os.environ.get('CXX', 'g++' if sys.platform == 'win32' else 'c++')
    flags = ['-O2', '-std=c++17', '-fPIC', '-ffp-contract=off', '-fno-fast-math',
             '-fno-builtin-sin', '-fno-builtin-cos', '-fno-builtin-sincos',
             '-I', sysconfig.get_paths()['include'], '-I', numpy.get_include()]
    env = dict(os.environ)
    if sys.platform == 'darwin':
        flags += ['-bundle', '-undefined', 'dynamic_lookup']
        # the Xcode toolchain on this laptop has an unaccepted licence
        env.setdefault('DEVELOPER_DIR', '/Library/Developer/CommandLineTools')
    else:
        flags += ['-shared']
    link = []
    if sys.platform == 'win32':
        # Link the active interpreter and keep MinGW runtime DLLs out of PATH requirements.
        link = ['-L', str(pathlib.Path(sys.base_prefix)/'libs'),
                '-lpython'+str(sys.version_info.major)+str(sys.version_info.minor),
                '-static']
    tmp = out.with_suffix('.tmp')
    cmd = [cxx, *flags, str(src), *link, '-o', str(tmp)]
    if verbose:
        print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)
    os.replace(tmp, out)  # new inode: running processes keep their loaded copy
    (HERE/'build-info.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        vendor_sha256=hashlib.sha256((HERE/'vendor_engine.cpp').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        python=sys.version, numpy=numpy.__version__, command=cmd,
        compiler=subprocess.check_output([cxx, '--version'], text=True).splitlines()[0]), indent=2)+'\n')
    return out


if __name__ == '__main__':
    print(build())
