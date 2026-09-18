"""ReplayRecorder-compatible disk-backed frame capture; no simulator changes.

Only the most recent frame stays in RAM. Original JSON replay schema and every
native frame are preserved. The journal remains on disk if publication fails.
"""
from pathlib import Path
import gzip
import json
import os
import tempfile
from recorder import ReplayRecorder as InMemoryRecorder, number


class FrameJournal:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='frames-', suffix='.jsonl.part', dir=directory)
        self.path = Path(name)
        self.stream = os.fdopen(fd, 'w', encoding='utf8')
        self.count = 0
        self.last = None
        self.dirty = False

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index == -1 and self.count:
            return self.last
        raise IndexError('Streaming recorder keeps only the latest frame in RAM')

    def append(self, frame):
        self.flush()
        self.last = frame
        self.count += 1
        self.dirty = True

    def flush(self):
        if self.dirty:
            json.dump(self.last, self.stream, separators=(',', ':'), allow_nan=False)
            self.stream.write('\n')
            self.stream.flush()
            self.dirty = False

    def copy_array(self, output):
        self.flush()
        output.write('[')
        with self.path.open() as source:
            for index, line in enumerate(source):
                if index:
                    output.write(',')
                output.write(line.rstrip('\n'))
        output.write(']')


class ReplayRecorder(InMemoryRecorder):
    def __init__(self, *args, spool_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        directory = spool_dir or os.environ.get('SURVIVAL_REPLAY_SPOOL_DIR')
        directory = directory or Path(__file__).resolve().parents[1] / 'results' / 'replay_spool'
        self.frames = FrameJournal(directory)
        self.meta['frame_storage'] = 'disk journal; every captured frame preserved'
        self.journal_meta = self.frames.path.with_suffix('.meta.part')
        self.journal_meta.write_text(json.dumps({'meta': self.meta, 'world': self.world}))

    def capture(self, *args, **kwargs):
        super().capture(*args, **kwargs)
        self.frames.flush()

    def save(self, path, *, reason='recording stopped', overwrite=False):
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        self.capture(force=True)
        summary = dict(duration=number(self.env.time), score=number(self.env.score),
                       frames=len(self.frames), reason=reason)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(temporary, 'wt', encoding='utf8') as output:
                output.write('{')
                for key, value in [('format','survival-replay'),('version',1),
                                   ('meta',self.meta),('world',self.world)]:
                    output.write(json.dumps(key) + ':')
                    json.dump(value, output, separators=(',', ':'), allow_nan=False)
                    output.write(',')
                output.write('"frames":')
                self.frames.copy_array(output)
                output.write(',"events":')
                json.dump(self.events, output, separators=(',', ':'), allow_nan=False)
                output.write(',"summary":')
                json.dump(summary, output, separators=(',', ':'), allow_nan=False)
                output.write('}')
            if overwrite:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
            self.frames.stream.close()
            self.frames.path.unlink()
            self.journal_meta.unlink()
        finally:
            temporary.unlink(missing_ok=True)
        return summary
