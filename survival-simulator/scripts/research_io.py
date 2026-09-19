"""Atomic research artifacts and storage accounting during concurrent writes."""
import json
import os
from pathlib import Path
import stat
import tempfile
import time


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                # Windows briefly denies replacement while another writer or
                # scanner holds the destination. Do not hide persistent denial.
                if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32) or attempt == 7:
                    raise
                time.sleep(.01*(attempt+1))
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode('utf-8'))


def directory_bytes(root):
    """Count regular files without a separate exists/stat race or symlink traversal.

    A temporary file can disappear after directory enumeration. That is normal;
    permission errors and other I/O failures must still be reported.
    """
    total, pending = 0, [Path(root)]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(metadata.st_mode):
                        total += metadata.st_size
        except FileNotFoundError:
            continue
    return total
