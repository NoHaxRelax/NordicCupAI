"""Small portable helpers. No GPU dependencies at import time."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def intersect(a, b):
    c = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return c if c[2] > c[0] and c[3] > c[1] else None


def covered(box, regions):
    """Exact rectangle-union coverage, including adjoining reviewed tiles."""
    cuts = sorted({box[0], box[2]} | {max(box[0], min(box[2], r[i])) for r in regions for i in (0, 2)})
    for left, right in zip(cuts, cuts[1:]):
        if right <= left:
            continue
        intervals = sorted((max(box[1], r[1]), min(box[3], r[3])) for r in regions
                           if r[0] <= left and r[2] >= right and r[3] > box[1] and r[1] < box[3])
        edge = box[1]
        for low, high in intervals:
            if low > edge:
                break
            edge = max(edge, high)
        if edge < box[3]:
            return False
    return True


def frame_split(frame, config):
    for split, key in [('train', 'validation_train_frames'), ('dev', 'validation_dev_frames')]:
        low, high = config[key]
        if low <= frame <= high:
            return split
    return None


def release_errors(manifest, release):
    errors = []
    for key in ['annotations_frozen', 'validation_training_permitted', 'launch_enabled']:
        if release.get(key) is not True:
            errors.append(key + ' is not confirmed')
    if release.get('source_fingerprint') != manifest['source_fingerprint']:
        errors.append('release does not name this exact annotation/config snapshot')
    return errors


def verify_dataset(root, manifest):
    """Verify every image and label before a GPU run, not only manifest identity."""
    seen = set()
    for row in manifest['records']:
        for key, hash_key in [('file', 'sha256'), ('label_file', 'label_sha256')]:
            if key not in row or row[key] in seen:
                continue
            seen.add(row[key])
            path = Path(root) / row[key]
            if sha(path) != row[hash_key]:
                raise ValueError('Dataset content changed: ' + row[key])
