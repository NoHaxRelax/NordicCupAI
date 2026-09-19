"""Restart a failed queue daemon without terminating its separately owned games."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from research_dispatch import write


def main():
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--worker-id', required=True)
    p.add_argument('--slots', type=int, default=30)
    a = p.parse_args()
    prefix = a.queue/('worker-'+a.worker_id)
    with prefix.with_suffix('.service.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = []
        while not (a.queue/'STOP').exists():
            with prefix.with_suffix('.v2.console.log').open('ab') as log:
                process = subprocess.Popen([sys.executable, '-B', '-u',
                    str(Path(__file__).with_name('research_dispatch_v2.py')), 'worker',
                    '--queue', str(a.queue), '--worker-id', a.worker_id, '--slots', str(a.slots)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                prefix.with_suffix('.pid').write_text(str(process.pid))
                write(prefix.with_suffix('.service.json'), dict(pid=os.getpid(), worker_pid=process.pid,
                    updated_at=time.time(), recent_failures=len(failures)))
                code = process.wait()
            if code == 0:
                return
            failures = [stamp for stamp in failures if time.time()-stamp < 600] + [time.time()]
            if len(failures) >= 5:
                write(prefix.with_suffix('.failed.json'), dict(at=time.time(), reason='Five queue-daemon failures in ten minutes'))
                (a.queue/'STOP').touch()
                return
            time.sleep(5)


if __name__ == '__main__':
    main()
