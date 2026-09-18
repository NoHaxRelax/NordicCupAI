"""Bounded high-resolution context follow-up on the existing mypc GPU."""
import ctypes, hashlib, json, os, subprocess, sys, tarfile, time, traceback
from pathlib import Path


def main(root):
    root=Path(root).resolve();old=root.parent/'scratch-bank-20260917-v1'
    out=root/'results';out.mkdir(exist_ok=False)
    os.environ.update(YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',YOLO_CONFIG_DIR=str(root/'settings'))
    (root/'settings').mkdir(exist_ok=True)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    def run(stage,args):
        (out/'status.json').write_text(json.dumps(dict(stage=stage,started=time.time())))
        with (out/(stage+'.out')).open('w') as stdout,(out/(stage+'.err')).open('w') as stderr:
            subprocess.run([sys.executable,'-u',*args],cwd=root,stdout=stdout,stderr=stderr,check=True)
    try:
        # Check again immediately before launch, without touching unrelated jobs.
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(used.strip().splitlines()[0])>2000:raise RuntimeError('GPU is occupied; refusing to overlap another training run')
        prior=old/'followup-v5-results/adapted'
        run('train',['train.py','--data',str(root/'data'),'--output',str(out/'trained'),'--epochs','96','--batch','8','--imgsz','960','--lr','.0005','--degrees','15','--amp','--continue-from',str(prior/'fit/weights/last.pt'),'--origin-completion',str(prior/'complete.json')])
        for zoom in [1,2]:
            run(f'eval-L{zoom}',['evaluate.py','--weights',str(out/'trained/fit/weights/last.pt'),'--fixture',str(old/'scratch-eval-v1'),'--output',str(out/f'eval-L{zoom}'),'--zoom',str(zoom)])
        (out/'status.json').write_text(json.dumps(dict(stage='complete',finished=time.time())))
        with tarfile.open(root/'results.tar.gz','w:gz') as archive:
            for path in out.rglob('*'):
                if path.is_file() and (path.suffix!='.pt' or path.name=='last.pt'):
                    archive.add(path,arcname=str(path.relative_to(root)))
            for name in ['train.py','evaluate.py','run_context_mypc.py','data/manifest.json']:
                archive.add(root/name,arcname=name)
        (root/'complete.json').write_text(json.dumps(dict(finished=time.time(),archive_sha256=hashlib.sha256((root/'results.tar.gz').read_bytes()).hexdigest())))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
