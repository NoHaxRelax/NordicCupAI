"""Compile fastsim._mirror in place: python mirror/build_mirror.py

Mirrors fastsim/build.py exactly, including -ffp-contract=off and the disabled sin/cos
builtins, because a shadow model that rounds differently from the engine it shadows is
worse than no shadow at all. The output lands next to fastsim/_engine so that
`from fastsim import _mirror` works with no path juggling.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import sysconfig

import numpy

HERE = pathlib.Path(__file__).resolve().parent
FASTSIM = HERE.parent / "fastsim"


def build(verbose=True):
    src = HERE / "_mirror.cpp"
    out = FASTSIM / ("_mirror" + sysconfig.get_config_var("EXT_SUFFIX"))
    cxx = os.environ.get("CXX", "g++" if sys.platform == "win32" else "c++")
    flags = [
        "-O2", "-std=c++17", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
        "-fno-builtin-sin", "-fno-builtin-cos", "-fno-builtin-sincos",
        "-I", str(FASTSIM),  # so #include "_engine.cpp" resolves
        "-I", sysconfig.get_paths()["include"], "-I", numpy.get_include(),
    ]
    env = dict(os.environ)
    if sys.platform == "darwin":
        flags += ["-bundle", "-undefined", "dynamic_lookup"]
        env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    else:
        flags += ["-shared"]
    link = []
    if sys.platform == "win32":
        link = ["-L", str(pathlib.Path(sys.base_prefix) / "libs"),
                "-lpython" + str(sys.version_info.major) + str(sys.version_info.minor),
                "-static"]
    tmp = out.with_suffix(".tmp")
    cmd = [cxx, *flags, str(src), *link, "-o", str(tmp)]
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    os.replace(tmp, out)
    (HERE / "build-info-mirror.json").write_text(json.dumps(dict(
        mirror_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        engine_source_sha256=hashlib.sha256((FASTSIM / "_engine.cpp").read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        python=sys.version, numpy=numpy.__version__, command=cmd,
        compiler=subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
    ), indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(build())
