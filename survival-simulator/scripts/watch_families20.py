"""Read progress from all ten pods. Run with --once or leave refreshing each minute."""
import argparse,concurrent.futures,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');a=ap.parse_args()
    dispatch=json.loads((ROOT/'docs/families20/dispatch.json').read_text())
    def check(p):
        remote=f"cd {dispatch['remote_root']} && tail -1 {p['family']}.log"
        try:
            r=subprocess.run(['ssh','-i',dispatch['ssh_key'],'-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=5','-p',str(p['port']),f"root@{p['host']}",remote],capture_output=True,text=True,timeout=12)
            return p['family'],r.stdout.strip()or r.stderr.strip()
        except subprocess.TimeoutExpired:return p['family'],'SSH timeout'
    while True:
        print(time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10)as pool:
            for family,status in pool.map(check,dispatch['assignments']):print(f'{family}: {status}',flush=True)
        if a.once:break
        time.sleep(60)
if __name__=='__main__':main()
