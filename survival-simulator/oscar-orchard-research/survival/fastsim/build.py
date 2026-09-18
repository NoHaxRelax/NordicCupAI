"""Compile the native engine in place: python fastsim/build.py

Needs only a C++17 compiler and the Python headers (no pybind11/setuptools).
-ffp-contract=off keeps every multiply and add separately rounded, like numpy.
The sin/cos builtins are disabled so the compiler cannot merge them into a
sincos call, which may round differently from numpy's separate libm calls.
"""
import os, subprocess, sys, sysconfig, pathlib
import numpy

HERE = pathlib.Path(__file__).resolve().parent


def build(verbose=True):
    src = HERE / '_engine.cpp'
    out = HERE / ('_engine' + sysconfig.get_config_var('EXT_SUFFIX'))
    cxx = os.environ.get('CXX', 'c++')
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
    tmp = out.with_suffix('.tmp')
    cmd = [cxx, *flags, str(src), '-o', str(tmp)]
    if verbose:
        print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)
    os.replace(tmp, out)  # new inode: running processes keep their loaded copy
    return out


if __name__ == '__main__':
    print(build())
