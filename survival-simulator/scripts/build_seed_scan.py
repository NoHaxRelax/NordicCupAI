"""Build a standalone scanner reusing the engine's CPython-compatible RNG."""
import pathlib
import subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
build=ROOT/'models/seed_shadow/build'
build.mkdir(exist_ok=True)
source=(ROOT/'fastsim/_shared_pysem.inc').read_text()
rng=source[source.index('struct PyRandom {'):source.index('// Python / numpy float semantics')]
(build/'seed_random.inc').write_text(rng)
subprocess.run(['g++','-O3','-std=c++17','-I',str(build),str(ROOT/'models/seed_shadow/scan.cpp'),'-o',str(build/'scan')],check=True)
print(build/'scan')
