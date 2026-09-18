"""Background detector processes for the asynchronous endpoint.

Kept out of example.py so that spawned workers importing this module do not
build a pool of their own. Each worker loads the configured detector once and
answers (sequence_id, frame_index, meta, png) tasks with detector rows.
"""
import atexit
import collections
import logging
import os
import queue
import threading
import time

import numpy as np


def worker_main(index, environ, tasks, results):
    """One detector process: decode a view, detect, report with the frame's identity."""
    import cv2
    os.environ.update(environ)
    os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')
    logging.basicConfig(level=logging.INFO)
    threads = int(environ.get('DRONE_WORKER_THREADS', '0')) or max(2, (os.cpu_count() or 8)//max(1, int(environ.get('DRONE_WORKERS', '4'))))
    threads = min(threads, 16)
    cv2.setNumThreads(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass
    from detectors import build_detector
    detector = build_detector()
    if hasattr(detector, 'restore_threads'):
        detector.restore_threads()
    cv2.setNumThreads(threads)  # Ultralytics resets OpenCV to one thread on import
    results.put(('ready', index, None, None, 0.))
    while True:
        task = tasks.get()
        if task is None:
            break
        sequence_id, frame_index, meta, png = task
        started = time.perf_counter()
        try:
            image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            rows = detector(image, meta)
            error = None
        except Exception as exc:  # report, never die
            rows, error = [], f'{type(exc).__name__}: {exc}'
        results.put((sequence_id, frame_index, rows, error, (time.perf_counter()-started)*1000))


class DetectorPool:
    """Bounded queue of views in front of N detector processes; results are drained by the caller."""
    def __init__(self, workers, backlog):
        import multiprocessing as mp
        context = mp.get_context('spawn')
        self.tasks = context.Queue()
        self.results = context.Queue()
        environ = {k: v for k, v in os.environ.items() if k.startswith('DRONE_') and k != 'DRONE_ASYNC'}
        self.processes = [context.Process(target=worker_main, args=(i, environ, self.tasks, self.results), daemon=True)
                          for i in range(workers)]
        for process in self.processes:
            process.start()
        self.workers = workers
        self.backlog = backlog
        self.pending = collections.deque()
        self.outstanding = 0
        self.dropped = 0
        self.lock = threading.Lock()
        self.ready = 0
        atexit.register(self.close)
        self.wait_ready(float(os.environ.get('DRONE_WORKER_READY_S', '900')))

    def wait_ready(self, timeout):
        """Block until every worker has loaded its detector (or died), so the endpoint never answers un-served."""
        deadline = time.monotonic()+timeout
        while self.ready < self.workers and time.monotonic() < deadline:
            self.drain()
            if self.alive() < self.workers-self.ready:
                break
            time.sleep(.5)
        logging.getLogger(__name__).info('detector pool: %d/%d workers ready, %d alive', self.ready, self.workers, self.alive())

    def submit(self, sequence_id, frame_index, meta, png):
        with self.lock:
            # Keep the freshest views: drop the oldest queued one when the backlog is full.
            if len(self.pending) >= self.backlog:
                self.pending.popleft(); self.dropped += 1
            self.pending.append((sequence_id, frame_index, meta, png))
            self._feed()

    def _feed(self):
        while self.pending and self.outstanding < self.workers:
            self.tasks.put(self.pending.popleft()); self.outstanding += 1

    def drain(self):
        """All finished detections so far, as (sequence_id, frame_index, rows, error, ms)."""
        done = []
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                break
            if item[0] == 'ready':
                self.ready += 1; continue
            done.append(item)
        with self.lock:
            self.outstanding = max(0, self.outstanding-len(done))
            self._feed()
        return done

    def alive(self):
        return sum(p.is_alive() for p in self.processes)

    def close(self):
        for _ in self.processes:
            try:
                self.tasks.put(None)
            except Exception:
                pass
