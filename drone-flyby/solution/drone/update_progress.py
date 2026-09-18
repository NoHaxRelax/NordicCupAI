#!/usr/bin/env python3
"""Atomically update the local dashboard progress receipt using real UTC."""
import argparse,json,os,tempfile
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--path',type=Path,required=True);p.add_argument('--through',type=int,required=True);p.add_argument('--message',required=True);a=p.parse_args()
j=json.loads(a.path.read_text()); now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
j['updated_at']=now;j['frames_reviewed']=list(range(5,a.through+1));j['message']=a.message;j['events'].append({'at':now,'message':a.message})
fd,tmp=tempfile.mkstemp(dir=a.path.parent,prefix='.progress.',suffix='.json');os.close(fd);Path(tmp).write_text(json.dumps(j,indent=2)+'\n');os.replace(tmp,a.path)
