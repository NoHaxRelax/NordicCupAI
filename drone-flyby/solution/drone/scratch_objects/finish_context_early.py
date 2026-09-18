"""Preserve an epoch-24 checkpoint, stop only this run, then evaluate it."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path


def main(root):
    root=Path(root).resolve();out=root/'early-24';out.mkdir(exist_ok=False)
    os.environ.update(YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(root/'settings'),OMP_NUM_THREADS='4')
    def status(**kw):(out/'status.json').write_text(json.dumps(kw))
    status(stage='waiting_for_epoch_24')
    try:
        import torch
        deadline=time.monotonic()+2400
        while time.monotonic()<deadline:
            progress=root/'results/trained/progress.json'
            p=json.loads(progress.read_text()) if progress.exists() else {}
            source=root/'results/trained/fit/weights/last.pt'
            if p.get('epoch',0)>=24 and source.exists():
                try:
                    before=source.stat();shutil.copy2(source,out/'last.pt');after=source.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):time.sleep(3);continue
                    checkpoint=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
                    epoch=checkpoint['epoch']+1;del checkpoint
                    if epoch<24:time.sleep(3);continue
                    # Resolve the owned training child by exact unique script path.
                    script=str(root/'train.py').replace("'","''")
                    cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*"+script+"*' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
                    found=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True).strip()
                    # The child is invoked as train.py relative to cwd by our runner.
                    if not found:
                        cmd="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*train.py*' -and $_.CommandLine -like '*scratch-context-20260918-v1*' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
                        found=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True).strip()
                    rows=json.loads(found) if found else [];rows=rows if isinstance(rows,list) else [rows]
                    pids={r['ProcessId']for r in rows};roots=[r for r in rows if r['ParentProcessId']not in pids]
                    if len(roots)!=1:raise RuntimeError('Expected one owned training process tree: '+str(rows))
                    subprocess.run(['taskkill','/PID',str(roots[0]['ProcessId']),'/T','/F'],check=True)
                    receipt=dict(epoch=epoch,checkpoint_sha256=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest(),stopped_process=rows[0],reason='User requested shorter training; checkpoint copied and loaded successfully before stopping',progress=p)
                    (out/'stop-receipt.json').write_text(json.dumps(receipt,indent=2));break
                except (EOFError,PermissionError):time.sleep(3)
            time.sleep(10)
        else:raise TimeoutError('No epoch-24 checkpoint in bounded waiting period')
        time.sleep(5)
        for zoom in [1,2]:
            status(stage=f'evaluate_L{zoom}',epoch=epoch)
            with (out/f'eval-L{zoom}.out').open('w') as stdout,(out/f'eval-L{zoom}.err').open('w') as stderr:
                subprocess.run([sys.executable,'-u',str(root/'evaluate.py'),'--weights',str(out/'last.pt'),'--fixture',str(root.parent/'scratch-bank-20260917-v1/scratch-eval-v1'),'--output',str(out/f'eval-L{zoom}'),'--zoom',str(zoom)],stdout=stdout,stderr=stderr,check=True)
        status(stage='complete',epoch=epoch)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
