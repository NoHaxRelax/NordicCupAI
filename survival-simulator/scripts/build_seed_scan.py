"""Build a standalone scanner reusing the engine's CPython-compatible RNG."""
import pathlib
import subprocess
import argparse
p=argparse.ArgumentParser();p.add_argument('--avx2',action='store_true');p.add_argument('--index',action='store_true');args=p.parse_args()
ROOT=pathlib.Path(__file__).resolve().parents[1]
build=ROOT/'models/seed_shadow/build'
build.mkdir(exist_ok=True)
source=(ROOT/'fastsim/_shared_pysem.inc').read_text()
rng=source[source.index('struct PyRandom {'):source.index('// Python / numpy float semantics')]
(build/'seed_random.inc').write_text(rng)
out=build/(('terrain-index' if args.index else 'scan')+('-avx2' if args.avx2 else ''))
subprocess.run(['g++','-O3','-std=c++17',*(['-mavx2'] if args.avx2 else []),'-I',str(build),str(ROOT/'models/seed_shadow'/('terrain_index.cpp' if args.index else 'scan.cpp')),'-o',str(out)],check=True)
print(out)
