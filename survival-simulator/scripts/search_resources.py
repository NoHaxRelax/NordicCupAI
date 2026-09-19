"""Read CPU/memory allocation without benchmarks or starting workloads."""
import math
import os
from pathlib import Path


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def _cgroups():
    """Include visible ancestors: a container may inherit a tighter parent quota."""
    root = Path('/sys/fs/cgroup')
    paths = {root}
    for line in _read('/proc/self/cgroup').splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers:
            candidate = root/relative.lstrip('/')
            if candidate.is_dir() and '..' not in candidate.parts:
                paths.update(p for p in (candidate, *candidate.parents) if p == root or root in p.parents)
    return sorted(paths)


def allocation(memory_per_worker_gb=2.):
    cpus = float(os.cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    memory_bytes = []
    for line in _read('/proc/meminfo').splitlines():
        if line.startswith('MemAvailable:'):
            memory_bytes.append(int(line.split()[1])*1024)
    for path in _cgroups():
        quota = _read(path/'cpu.max').split()
        if len(quota) == 2 and quota[0] != 'max' and int(quota[1]) > 0:
            cpus = min(cpus, int(quota[0])/int(quota[1]))
        limit, used = _read(path/'memory.max'), _read(path/'memory.current')
        if limit.isdigit() and used.isdigit():
            memory_bytes.append(max(0, int(limit)-int(used)))
    # Conventional cgroup-v1 mounts, used by some older container hosts.
    for mount in ('cpu', 'cpu,cpuacct'):
        base = Path('/sys/fs/cgroup')/mount
        quota, period = _read(base/'cpu.cfs_quota_us'), _read(base/'cpu.cfs_period_us')
        if quota and period and int(quota) > 0 and int(period) > 0:
            cpus = min(cpus, int(quota)/int(period))
    base = Path('/sys/fs/cgroup/memory')
    limit, used = _read(base/'memory.limit_in_bytes'), _read(base/'memory.usage_in_bytes')
    if limit.isdigit() and used.isdigit() and int(limit) < 2**60:
        memory_bytes.append(max(0, int(limit)-int(used)))
    if os.name == 'nt' and not memory_bytes:
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total', 'available', 'page_total', 'page_available',
                           'virtual_total', 'virtual_available', 'extended')]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory_bytes.append(status.available)
    memory_gb = min(memory_bytes)/2**30 if memory_bytes else None
    cpu_slots = max(1, math.floor(cpus)-(2 if cpus >= 8 else 1))
    # Two GB per game is a planning allowance, not a measured memory bound.
    memory_slots = max(1, math.floor((memory_gb-4)/memory_per_worker_gb)) if memory_gb is not None else 1
    model = next((line.split(':', 1)[1].strip() for line in _read('/proc/cpuinfo').splitlines()
                  if line.startswith('model name')), 'unknown')
    return dict(effective_vcpus=cpus, available_memory_gib=memory_gb, cpu_model=model,
                memory_per_worker_gib=memory_per_worker_gb, suggested_workers=min(cpu_slots, memory_slots),
                cpu_slots=cpu_slots, memory_slots=memory_slots)
