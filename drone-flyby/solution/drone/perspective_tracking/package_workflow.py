"""Build a small standalone runtime bundle without weights or test data."""
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root/'artifacts/drone-revisit-workflow'
    directory.mkdir(parents=True, exist_ok=True)
    files = {'drone/__init__.py': b'',
             'README.md': (root/'docs/drone-revisit-workflow.md').read_bytes(),
             'pyproject.toml': b'''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "nordic-drone-tracker"
version = "0.2.0"
description = "Experimental drone perspective tracking with automatic revisits"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24,<3", "opencv-python>=4.8,<5"]

[project.optional-dependencies]
detector = ["ultralytics>=8.4,<9"]

[project.scripts]
drone-tracker = "drone.perspective_tracking.experiment:main"

[tool.setuptools.packages.find]
include = ["drone", "drone.*"]
'''}
    for filename in ('__init__.py', 'motion.py', 'tracker.py', 'revisit.py', 'workflow.py', 'experiment.py'):
        files['drone/perspective_tracking/'+filename] = (Path(__file__).parent/filename).read_bytes()
    files['MANIFEST.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                                     for name, data in sorted(files.items())).encode()
    path = directory/'drone-tracker-v0.2.zip'
    with ZipFile(path, 'w', compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo('drone-tracker-v0.2/'+name, date_time=(2026, 9, 17, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.zip.sha256').write_text(f'{digest}  {path.name}\n')
    print(path)
    print(digest)


if __name__ == '__main__':
    main()
