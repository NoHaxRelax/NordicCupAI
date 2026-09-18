"""Graceful task stop for quota, low system RAM, or low disk space."""
from pathlib import Path
import sys,time,json,shutil,importlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/intake_validation'))
import usage
while not usage.STOP.exists():
 importlib.reload(usage)
 quota=usage.status()
 values={row.split(':')[0]:int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith(('MemTotal:','MemAvailable:'))}
 free=shutil.disk_usage(ROOT).free
 if values['MemAvailable']<6*1024**3 or free<20*1024**3:
  usage.STOP.write_text(json.dumps({'reason':'resource headroom guard','memory':values,'disk_free_bytes':free})+'\n')
  print('Resource headroom guard stopped new work; running loops will save partials.',flush=True)
  break
 if quota.get('stop'):break
 time.sleep(3)
