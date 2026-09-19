"""Give queued background research lower CPU priority than main evaluations."""
import argparse
import os
from pathlib import Path
import time

from research_dispatch import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--worker-id', required=True)
    args = p.parse_args()
    while not (args.queue/'STOP').exists():
        parents, background = {}, set()
        for path in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(path.name)
                parents[pid] = int((path/'stat').read_text().rsplit(')', 1)[1].split()[1])
                command = (path/'cmdline').read_bytes().split(b'\0')
                if b'case' in command and any(c.endswith(b'/research_dispatch.py') for c in command):
                    i = command.index(b'--job')
                    job = Path(os.fsdecode(command[i+1]))
                    if job.parent == args.queue/'jobs' and int(job.name.split('-', 1)[0]) >= 20:
                        background.add(pid)
            except (OSError, ValueError, IndexError):
                continue
        for _ in range(8):
            more = {pid for pid, parent in parents.items() if parent in background} - background
            if not more:
                break
            background.update(more)
        for pid in background:
            try:
                os.setpriority(os.PRIO_PROCESS, pid, 8)
            except ProcessLookupError:
                pass
        write(args.queue/'priorities'/f'{args.worker_id}.json',
              dict(updated_at=time.time(), pid=os.getpid(), background_processes=len(background), nice=8))
        time.sleep(5)


if __name__ == '__main__':
    main()
