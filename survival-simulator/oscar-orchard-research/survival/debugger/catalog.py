"""Discover complete local replays, independent of the experiment's save folder.

The live server uses this catalog; no research files are moved or rewritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent
SURVIVAL = ROOT.parent
EXCLUDED = {'.venv', 'vendor', '.git', '__pycache__', 'node_modules'}


def signature(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def created_at_timestamp(value):
    """Return an epoch timestamp for ISO 8601 replay metadata."""
    if not isinstance(value, str):
        return float('-inf')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return float('-inf')


def read_replay(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def replay_digest(payload):
    """Hash canonical JSON using the C encoder, bounded to one frame at a time."""
    hasher = hashlib.sha256()
    def add(value):
        hasher.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8'))
    hasher.update(b'{')
    for index, key in enumerate(sorted(payload)):
        if index:
            hasher.update(b',')
        add(key)
        hasher.update(b':')
        if key == 'frames':
            hasher.update(b'[')
            for frame_index, frame in enumerate(payload[key]):
                if frame_index:
                    hasher.update(b',')
                add(frame)
            hasher.update(b']')
        else:
            add(payload[key])
    hasher.update(b'}')
    return hasher.hexdigest()[:32]


def inspect_replay(data):
    """Validate enough of the format to avoid offering incomplete/summary files."""
    if not isinstance(data, dict) or data.get('format') != 'survival-replay':
        return None
    if data.get('version') != 1:
        raise ValueError('Unsupported replay version')
    world = data.get('world', {})
    if not isinstance(world, dict) or any(not isinstance(world.get(k), (int, float)) or not math.isfinite(world[k]) or world[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Missing valid world dimensions')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('No recorded frames')
    last = -math.inf
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get('t'), (int, float)) or not math.isfinite(frame['t']) or frame['t'] < last:
            raise ValueError('Invalid frame chronology')
        if any(not isinstance(frame.get(k), list) for k in ('agents', 'predators', 'fruits', 'trees')):
            raise ValueError('Missing frame entities')
        last = frame['t']
    summary = data.get('summary')
    if not isinstance(summary, dict) or not isinstance(summary.get('duration'), (int, float)) or not math.isfinite(summary['duration']):
        raise ValueError('No completed recording summary')
    if summary.get('frames') != len(frames) or abs(summary['duration'] - last) > .001:
        raise ValueError('Incomplete final frame or summary')
    meta = data.get('meta', {})
    if not isinstance(meta, dict):
        raise ValueError('Invalid replay metadata')
    # Catalog copies may add a title/notes, but identical simulation data should
    # occupy one option. created_at preserves genuinely separate identical runs.
    payload = {'created_at': meta.get('created_at'), 'world': world,
               'frames': frames, 'events': data.get('events', []), 'summary': summary}
    return dict(id=replay_digest(payload), title=meta.get('title') or 'Untitled run',
                description=meta.get('notes', ''), duration=last, frames=len(frames),
                seed=meta.get('seed'), policy=meta.get('policy'),
                renderer='native' if all(f.get('native_image') for f in frames) else 'state',
                created_at=meta.get('created_at'), reason=summary.get('reason', ''))


class ReplayCatalog:
    def __init__(self, survival=SURVIVAL, debugger=None):
        self.survival = Path(survival).resolve()
        self.debugger = Path(debugger).resolve() if debugger else self.survival / 'debugger'
        self.cache = {}
        self.paths = {}
        self.lock = threading.RLock()
        self.scan_lock = threading.Lock()
        self.manifest = {'recordings': [], 'stats': {'scanning': True}}
        self.stop_event = threading.Event()

    def candidates(self):
        # A single project-scoped search also catches recordings outside results/.
        for directory, subdirs, names in os.walk(self.survival, followlinks=False):
            subdirs[:] = [name for name in subdirs if name not in EXCLUDED]
            if Path(directory) == self.debugger:
                subdirs[:] = [name for name in subdirs if name != 'tests']
            for name in names:
                if not (name.endswith('.json') or name.endswith('.json.gz')):
                    continue
                path = Path(directory) / name
                if path.is_file() and path.resolve().is_relative_to(self.survival):
                    yield path

    def curated(self):
        try:
            manifest = json.loads((self.debugger / 'recordings/manifest.json').read_text())
            entries = manifest['recordings']
            return {str((self.debugger / row['file']).resolve()): (i, row)
                    for i, row in enumerate(entries) if isinstance(row, dict) and isinstance(row.get('file'), str)
                    and (self.debugger / row['file']).resolve().is_relative_to(self.survival)}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def refresh(self):
        with self.scan_lock:
            curated = self.curated()
            files = sorted(self.candidates(), key=lambda p: (str(p.resolve()) not in curated, str(p)))
            seen = set()
            groups = {}
            errors = []
            non_replays = 0
            for path in files:
                key = str(path.resolve())
                seen.add(key)
                try:
                    sig = signature(path)
                    cached = self.cache.get(key)
                    if cached and cached[0] == sig:
                        info, error = cached[1:]
                    else:
                        info = error = None
                        try:
                            data = read_replay(path)
                            info = inspect_replay(data)
                        except (OSError, ValueError, TypeError, EOFError, OverflowError, KeyError) as exc:
                            error = str(exc)
                        if signature(path) != sig:
                            # Writer is still active. Retry after its next atomic
                            # save or modification; never publish a partial file.
                            continue
                        self.cache[key] = (sig, info, error)
                    if error:
                        errors.append({'source': str(path.relative_to(self.survival)), 'error': error})
                        continue
                    if info is None:
                        non_replays += 1
                        continue
                    source = str(path.relative_to(self.survival))
                    identifier = info['id']
                    if identifier in groups:
                        groups[identifier]['sources'].append(source)
                        continue
                    entry = dict(info, source=source, sources=[source])
                    if key in curated:
                        order, old = curated[key]
                        entry.update(title=old.get('title') or entry['title'], description=old.get('description') or entry['description'])
                        group = 'Original demonstrations'
                    else:
                        order = 100000
                        parts = path.relative_to(self.survival).parts
                        group = parts[1].replace('_', ' ').title() if len(parts) > 2 and parts[0] in ('results', 'research') else 'Other completed runs'
                    entry.update(group=group, file=f'replays/{identifier}' + ('.json.gz' if path.suffix == '.gz' else '.json'))
                    groups[identifier] = dict(entry=entry, sources=entry['sources'], path=path.resolve(), signature=sig, order=order)
                except (OSError, ValueError) as exc:
                    errors.append({'source': str(path.relative_to(self.survival)), 'error': str(exc)})
            self.cache = {key: value for key, value in self.cache.items() if key in seen}
            ordered = sorted(
                groups.values(),
                key=lambda row: (
                    -created_at_timestamp(row['entry'].get('created_at')),
                    row['entry']['source'],
                ),
            )
            # Equal descriptive titles are common across seeds and versions.
            counts = {}
            for row in ordered:
                label = row['entry']['title']
                counts[label] = counts.get(label, 0) + 1
            for row in ordered:
                entry = row['entry']
                if counts[entry['title']] > 1:
                    entry['title'] += ' · ' + Path(entry['source']).name
            manifest = dict(recordings=[row['entry'] for row in ordered], generated_at=time.time(),
                            stats=dict(scanning=False, recordings=len(ordered), replay_files=sum(len(row['sources']) for row in ordered),
                                       other_json_files=non_replays, invalid_files=len(errors)), errors=errors)
            with self.lock:
                self.manifest = manifest
                self.paths = {row['entry']['file']: (row['path'], row['signature']) for row in ordered}
            return manifest

    def snapshot(self):
        with self.lock:
            return self.manifest

    def resolve(self, route):
        with self.lock:
            item = self.paths.get(route)
        if item is None:
            return None
        path, sig = item
        try:
            if path.resolve().is_relative_to(self.survival) and signature(path) == sig:
                return path
        except OSError:
            pass
        return None

    def start(self, interval=5):
        def watch():
            while not self.stop_event.is_set():
                try:
                    self.refresh()
                except Exception as exc:
                    print(f'Catalog scan failed; keeping last complete list: {exc}', flush=True)
                self.stop_event.wait(interval)
        thread = threading.Thread(target=watch, name='replay-catalog', daemon=True)
        thread.start()
        return thread


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional inventory JSON file (not the curated manifest).')
    args = parser.parse_args()
    catalog = ReplayCatalog()
    manifest = catalog.refresh()
    if args.output:
        args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['stats'], indent=2))
    for row in manifest['recordings']:
        print(f"{row['group']}: {row['title']} [{row['duration']:g}s] <- {row['source']}")
    for error in manifest['errors']:
        print('Skipped incomplete/invalid JSON:', error['source'], error['error'])
