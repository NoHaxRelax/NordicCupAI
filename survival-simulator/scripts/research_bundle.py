"""Package the current source, including untracked edits, without local secrets/runs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_support import digest, manifest


def build(target):
    target = Path(target)
    if target.exists():
        raise ValueError('Use a new bundle path')
    files = manifest(ROOT, native=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, expected in files.items():
            data = (ROOT/name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f'Source changed during packaging: {name}')
            entry = tarfile.TarInfo('NordicCupAI/survival-simulator/'+name)
            entry.size, entry.mode = len(data), 0o755 if name.endswith('.sh') else 0o644
            archive.addfile(entry, io.BytesIO(data))
        data = json.dumps(dict(snapshot=digest(files), files=files), indent=2).encode()
        entry = tarfile.TarInfo('NordicCupAI/research-source-manifest.json')
        entry.size, entry.mode = len(data), 0o644
        archive.addfile(entry, io.BytesIO(data))
    with tarfile.open(temporary) as archive:
        for name, checksum in files.items():
            data = archive.extractfile('NordicCupAI/survival-simulator/'+name).read()
            if hashlib.sha256(data).hexdigest() != checksum:
                raise RuntimeError(f'Bundle verification failed: {name}')
    temporary.replace(target)
    report = dict(bundle=str(target.resolve()), source_snapshot=digest(files), files=len(files),
                  bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_name(target.name+'.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2))


if __name__ == '__main__':
    main()
