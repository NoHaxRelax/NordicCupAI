"""Process pool for the per-class expert stage: the experts' per-candidate Python work holds the GIL, so a thread
pool runs the 16 classes one after another. Each worker process builds the same experts (CPU only) once; per view the
delivered/upsampled image is shared through one shared-memory block and every class runs in some worker. Same code,
same inputs, so outputs are identical to the in-process call.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context, shared_memory

import numpy as np

_W = {}  # worker state: experts, attached shared-memory blocks


def _init(project, bank, gates, classes, sift):
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
    _W.update(experts=experts, shm={})


def _image(shm_name, shape):
    block = _W['shm'].get(shm_name)
    if block is None:
        for old in _W['shm'].values():
            old.close()
        _W['shm'].clear()
        block = _W['shm'][shm_name] = shared_memory.SharedMemory(name=shm_name)
    return np.ndarray(shape, np.uint8, buffer=block.buf)


def _task(name, shm_name, shape, scale, zoom, proposals):
    image = _image(shm_name, shape)
    expert, family = _W['experts'][name]
    try:
        out = expert.detect(image, scale, zoom, explain=False, proposals=proposals) if proposals is not None else expert.detect(image, scale, zoom, explain=False)
        rows = out[family] if isinstance(out, dict) else out[0]
        return [dict(r, expert=name) for r in rows]
    except Exception as error:
        print(f'expert {name} failed: {error}', file=sys.stderr)
        return []


class ExpertProcessPool:
    def __init__(self, processes, project, bank, gates, classes, sift=False, cost_order=None):
        self.pool = ProcessPoolExecutor(max_workers=int(processes), mp_context=get_context('spawn'),
                                        initializer=_init, initargs=(str(project), str(bank), str(gates) if gates else None, list(classes), bool(sift)))
        self.block = None
        self.order = list(cost_order or classes)  # submit the slowest classes first for load balance

    def _share(self, image):
        need = image.nbytes
        if self.block is None or self.block.size < need:
            if self.block is not None:
                self.block.close(); self.block.unlink()
            self.block = shared_memory.SharedMemory(create=True, size=max(need, 3840 * 2160 * 3))
        np.ndarray(image.shape, np.uint8, buffer=self.block.buf)[...] = image
        return self.block.name

    def run(self, names, image, scale, zoom, proposals):
        image = np.ascontiguousarray(image)
        shm_name = self._share(image)
        names = sorted(names, key=lambda n: self.order.index(n) if n in self.order else len(self.order))
        futures = {n: self.pool.submit(_task, n, shm_name, image.shape, scale, zoom, proposals.get(n)) for n in names}
        return {n: f.result() for n, f in futures.items()}

    def close(self):
        self.pool.shutdown(cancel_futures=True)
        if self.block is not None:
            self.block.close(); self.block.unlink(); self.block = None
