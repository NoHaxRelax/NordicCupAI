"""Bounded, lossy telemetry: disk and JSON work stays outside the policy process."""
import atexit
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing as mp
import os
from pathlib import Path
from queue import Full


def _writer(path, queue):
    try:
        os.nice(10)
    except OSError:
        pass
    logging.raiseExceptions = False
    handlers = {}
    try:
        for kind, suffix, size, backups in [('metrics', '', 32 << 20, 3),
                                           ('request', '.requests', 128 << 20, 7)]:
            dest = Path(path + suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(dest, maxBytes=size, backupCount=backups)
            handler.setFormatter(logging.Formatter('%(message)s'))
            handlers[kind] = handler
        while True:
            item = queue.get()
            if item is None:
                break
            kind, row, chunks = item
            try:
                if kind == 'request':
                    row['request'] = json.loads(b''.join(chunks))
                text = json.dumps(row, separators=(',', ':'), allow_nan=False)
                handlers[kind].emit(logging.LogRecord('telemetry', logging.INFO, '', 0,
                                                      text, (), None))
            except Exception:
                # Diagnostic failures must never take down the policy process.
                continue
    finally:
        for handler in handlers.values():
            handler.close()


class Telemetry:
    max_request_bytes = 512 << 10

    def __init__(self, path):
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue(maxsize=64)
        self.queue.cancel_join_thread()
        self.process = ctx.Process(target=_writer, args=(str(path), self.queue), daemon=True)
        self.process.start()
        self.enqueued = self.dropped = self.oversized = 0
        atexit.register(self.close)

    def emit(self, row, chunks=None):
        try:
            self.queue.put_nowait(('request' if chunks is not None else 'metrics', row, chunks))
            self.enqueued += 1
            return True
        except (Full, ValueError, OSError):
            self.dropped += 1
            return False

    def status(self):
        return dict(writer_alive=self.process.is_alive(), enqueued=self.enqueued,
                    dropped=self.dropped, oversized_requests=self.oversized,
                    max_request_bytes=self.max_request_bytes, queue_capacity=64)

    def close(self):
        try:
            self.queue.put_nowait(None)
        except (Full, ValueError, OSError):
            pass
