"""One serial, bounded follow-up on the existing RTX4080. No additional rental."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

def main(root):
    ROOT=Path(root).resolve()
    CODE=ROOT/'scratch-followup-v5'
    FIRST=ROOT/'run-amp-v3'
    OUT=ROOT/'followup-v5-results'
    OUT.mkdir(exist_ok=False)
    os.environ.update(YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',YOLO_CONFIG_DIR=str(ROOT/'settings'))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    def run(name,args):
        (OUT/'status.json').write_text(json.dumps(dict(stage=name,started=time.time())))
        with (OUT/(name+'.out')).open('w') as out,(OUT/(name+'.err')).open('w') as err:
            subprocess.run([sys.executable,'-u',*args],cwd=CODE,stdout=out,stderr=err,check=True)

    def evaluate(weights,prefix):
        for zoom in [0,1,2]:
            run(prefix+f'-L{zoom}',['-m','scratch_objects.evaluate','--weights',str(weights),'--fixture',str(ROOT/'scratch-eval-v1'),
                 '--output',str(OUT/(prefix+f'-L{zoom}')),'--zoom',str(zoom)])

    try:
        deadline=time.monotonic()+3600
        while not (FIRST/'complete.json').exists():
            if (ROOT/'train-amp-v3.failure.txt').exists():raise RuntimeError('Initial training failed; no follow-up launched')
            if time.monotonic()>deadline:raise TimeoutError('Initial run did not finish in bounded wait')
            time.sleep(15)
        weights=FIRST/'fit/weights/last.pt'
        expected=json.loads((FIRST/'complete.json').read_text())['last_sha256']
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=expected:raise ValueError('Initial checkpoint identity mismatch')
        evaluate(weights,'synthetic-final')
        hard=ROOT/'scratch-bank-clean-hard-v5'
        run('mine-training-negatives',['-m','scratch_objects.mine_negatives','--data',str(ROOT/'scratch-bank-clean-native-v4'),
             '--weights',str(weights),'--output',str(hard)])
        trained=OUT/'adapted'
        run('adapt-scratch-features',['-m','scratch_objects.train','--data',str(hard),'--output',str(trained),'--epochs','48','--batch','16','--amp',
             '--continue-from',str(weights),'--origin-completion',str(FIRST/'complete.json')])
        evaluate(trained/'fit/weights/last.pt','adapted-final')
        (OUT/'complete.json').write_text(json.dumps(dict(finished=time.time(),status='completed')))
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    main(parser.parse_args().root)
