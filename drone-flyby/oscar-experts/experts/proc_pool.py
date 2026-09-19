"""Process pool for the per-class expert stage: the experts' per-candidate Python work holds the GIL, so a thread
pool runs the 16 classes one after another. Classes are PINNED to worker processes (each class always runs in the same
worker, which builds only its own experts), so every worker's pose caches, GPU kernel spectra and FFT plans stay warm;
with a shared queue each view hit cold caches in whichever worker picked a class up. Per view the image is shared
through one shared-memory block per scene. Same code, same inputs: outputs are identical to the in-process call.
"""
import os
import sys
import time
from contextlib import contextmanager
from multiprocessing import get_context, shared_memory

import numpy as np


def _worker(conn, project, bank, gates, classes, sift, index=0):
    os.environ['DRONE_EXPERT_WORKER_INDEX'] = str(index)  # gpu_window picks this worker's GPU from a device list
    import cv2
    cv2.setNumThreads(1)
    if project not in sys.path:
        sys.path.insert(0, project)
    from drone.experts.registry import make_expert, families, load_gates
    table = load_gates(gates) if gates else None
    experts = {}
    for name in classes:
        try:
            experts[name] = (make_expert(name, bank, table), families(name)[0])
        except Exception as error:
            print(f'expert {name} unavailable in worker: {error}', file=sys.stderr)
    if not sift:
        for expert, _ in experts.values():
            if hasattr(expert, 'sift'):
                expert.sift = None
    blocks = {}
    conn.send('ready')
    trace = os.environ.get('DRONE_EXPERT_POOL_TRACE') == '1'
    while True:
        msg = conn.recv()  # one message per view: this worker's whole task list, so sends never interleave (no deadlock)
        if msg is None:
            break
        results = []
        for tag, name, shm_name, shape, scale, zoom, proposals in msg:
            started = time.time()
            block = blocks.get(shm_name)
            if block is None:
                block = blocks[shm_name] = shared_memory.SharedMemory(name=shm_name)
            image = np.ndarray(shape, np.uint8, buffer=block.buf)
            rows = []
            if name in experts:
                expert, family = experts[name]
                try:
                    out = expert.detect(image, scale, zoom, explain=False, proposals=proposals) if proposals is not None else expert.detect(image, scale, zoom, explain=False)
                    rows = [dict(r, expert=name) for r in (out[family] if isinstance(out, dict) else out[0])]
                except Exception as error:
                    print(f'expert {name} failed: {error}', file=sys.stderr)
            if trace:
                print(f'POOLTRACE {name} pid={os.getpid()} run={time.time() - started:.3f}', file=sys.stderr, flush=True)
            results.append((tag, rows))
        conn.send(results)
    for block in blocks.values():
        block.close()


@contextmanager
def _hidden_main():
    """spawn re-runs the caller's __main__ in every worker when __main__ has a __file__ or __spec__ (api.py -> example.py
    builds a detector at import -> its pool spawns workers -> ...). Hide both while workers are started."""
    main = sys.modules.get('__main__')
    saved = {k: getattr(main, k) for k in ('__file__', '__spec__') if main is not None and hasattr(main, k)}
    try:
        if '__spec__' in saved:
            main.__spec__ = None
        if '__file__' in saved:
            del main.__file__
        yield
    finally:
        for k, v in saved.items():
            setattr(main, k, v)


class ExpertProcessPool:
    def __init__(self, processes, project, bank, gates, classes, sift=False, cost_order=None):
        classes = list(classes)
        order = [c for c in (cost_order or []) if c in classes] + [c for c in classes if c not in (cost_order or [])]
        n = max(1, min(int(processes), len(classes)))
        # slowest classes first, round-robin: with >= 16 workers every class has its own process
        self.assign = {c: i % n for i, c in enumerate(order)}
        ctx = get_context('spawn')
        self.conns, self.procs = [], []
        with _hidden_main():
            for w in range(n):
                mine = [c for c in order if self.assign[c] == w]
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, str(project), str(bank), str(gates) if gates else None, mine, bool(sift), w), daemon=True)
                p.start()
                self.conns.append(parent); self.procs.append(p)
        for c in self.conns:
            if c.recv() != 'ready':
                raise RuntimeError('expert worker failed to start')
        self.blocks = {}
        self.order = order

    def _share(self, image, slot=0):
        need = image.nbytes
        block = self.blocks.get(slot)
        if block is None or block.size < need:
            if block is not None:
                block.close(); block.unlink()
            block = self.blocks[slot] = shared_memory.SharedMemory(create=True, size=max(need, 3840 * 2160 * 3))
        np.ndarray(image.shape, np.uint8, buffer=block.buf)[...] = image
        return block.name

    def run(self, names, image, scale, zoom, proposals):
        return self.run_jobs([(names, image, scale, proposals)], zoom)

    def run_jobs(self, jobs, zoom):
        """jobs: list of (class names, image, pixels per native pixel, proposals by class); one shared block per job."""
        shared = []
        for slot, (names, image, scale, proposals) in enumerate(jobs):
            image = np.ascontiguousarray(image)
            shared.append((self._share(image, slot), image.shape))
        tasks = {}
        for n in self.order:  # slowest first within each worker
            for slot, (names, image, scale, proposals) in enumerate(jobs):
                if n in names:
                    tasks.setdefault(self.assign[n], []).append((n, n, shared[slot][0], shared[slot][1], scale, zoom, proposals.get(n)))
        for w, items in tasks.items():
            self.conns[w].send(items)
        out = {}
        for w in tasks:
            for tag, rows in self.conns[w].recv():
                out[tag] = rows
        return out

    def run_stream(self, scenes, zoom, produce, proposer_classes):
        """scenes: list of (class names, image, pixels per native pixel). produce(callback) runs the shared proposer and
        calls callback(name, proposals) as each class's proposals are complete. A worker's message is sent as soon as all
        of its classes in this view are ready, so experts run while the GPU is still proposing for other classes; classes
        without shared proposals (hangar, blob experts) are sent at once. Results are identical to run_jobs."""
        shared = []
        for slot, (names, image, scale) in enumerate(scenes):
            image = np.ascontiguousarray(image)
            shared.append((self._share(image, slot), image.shape))
        slot_of = {n: slot for slot, (names, _, _) in enumerate(scenes) for n in names}
        need = {}
        for n in slot_of:
            need.setdefault(self.assign[n], set()).add(n)
        ready = {w: {} for w in need}
        rank = {n: i for i, n in enumerate(self.order)}

        def callback(name, proposals):
            if name not in slot_of:
                return
            w, slot = self.assign[name], slot_of[name]
            ready[w][name] = (name, name, shared[slot][0], shared[slot][1], scenes[slot][2], zoom, proposals)
            if len(ready[w]) == len(need[w]):
                self.conns[w].send([ready[w][n] for n in sorted(ready[w], key=lambda n: rank.get(n, len(rank)))])
        for n in slot_of:
            if n not in proposer_classes:
                callback(n, None)
        produce(callback)
        for n in slot_of:  # anything the proposer did not report runs on its own proposals
            w = self.assign[n]
            if n not in ready[w]:
                callback(n, None)
        out = {}
        for w in need:
            for tag, rows in self.conns[w].recv():
                out[tag] = rows
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
        for block in self.blocks.values():
            block.close(); block.unlink()
        self.blocks = {}
