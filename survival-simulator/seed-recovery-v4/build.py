"""Build the frozen V4 dependency closure on its destination Linux CPU."""
from pathlib import Path
import platform, shlex, subprocess, sys, sysconfig
if platform.system() != 'Linux' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Use Linux/WSL and CPython 3.12 with development headers.')
import numpy
if numpy.__version__ != '2.3.5':
    raise SystemExit('This checkpoint requires numpy==2.3.5; revalidate parity before upgrading.')
root = Path(__file__).resolve().parent
src = root/'survival/research/seed_inference'
flags = ['g++', '-O3', '-std=c++17', '-march=native', '-ffp-contract=off', '-pthread']
http = shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'cpp-httplib'], text=True))
includes = ['-I'+sysconfig.get_path('include'), '-I'+numpy.get_include()]
links = ['-L'+sysconfig.get_config_var('LIBDIR'), '-lpython3.12', '-ldl', '-lm'] + http
for name in ['terrain_filter', 'stream_integration_filter']:
    subprocess.run(flags + [str(src/(name+'.cpp')), '-o', str(src/name)], check=True)
for name in ['paced_native_v4', 'paced_native_mode144', 'streaming_verification/coordinator', 'stream_integration_test', 'live_game_client']:
    subprocess.run(flags + includes + [str(src/(name+'.cpp')), '-o', str(src/name)] + links, check=True)
print('Built V4 and diagnostic tools. Run python preflight.py --native next.')
