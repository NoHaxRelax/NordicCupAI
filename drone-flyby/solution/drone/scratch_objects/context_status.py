"""Read progress of the isolated context training job (run on mypc)."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result={}
    for key,rel in [('stage','results/status.json'),('training','results/trained/progress.json'),('initialization','results/trained/continuation-verified.json'),('complete','complete.json')]:
        path=a.root/rel
        result[key]=json.loads(path.read_text()) if path.exists() else None
    error=a.root/'results/failure.txt'
    if error.exists():result['failure']=error.read_text()[-4000:]
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
