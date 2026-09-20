"""Compile the private native engine+policy fork in place: python nightsim/build.py
Same flags as fastsim/build.py (bit-identical engine arithmetic); module name _nengine."""
import os, subprocess, sys, sysconfig, pathlib
import numpy
HERE = pathlib.Path(__file__).resolve().parent

def build(verbose=True, opt='-O2'):
    src = HERE / '_nengine.cpp'
    out = HERE / ('_nengine' + sysconfig.get_config_var('EXT_SUFFIX'))
    cxx = os.environ.get('CXX', 'c++')
    flags = [opt, '-std=c++17', '-fPIC', '-ffp-contract=off', '-fno-fast-math',
             '-fno-builtin-sin', '-fno-builtin-cos', '-fno-builtin-sincos',
             '-I', sysconfig.get_paths()['include'], '-I', numpy.get_include()]
    env = dict(os.environ)
    if sys.platform == 'darwin':
        flags += ['-bundle', '-undefined', 'dynamic_lookup']
        env.setdefault('DEVELOPER_DIR', '/Library/Developer/CommandLineTools')
    else:
        flags += ['-shared']
    tmp = out.with_suffix('.tmp')
    cmd = [cxx, *flags, str(src), '-o', str(tmp)]
    if verbose: print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    os.replace(tmp, out)
    return out

if __name__ == '__main__':
    print(build())
