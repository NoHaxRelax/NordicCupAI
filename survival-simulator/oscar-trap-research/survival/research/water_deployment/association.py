"""Prevent averaging a second station's predator into the local track."""
from common import *
from run import run

rows=[]
prior=json.loads((OUT/'twins.json').read_text())['data']
for old in prior:
    c=dict(old['case'],association_gate=True)
    r=run(c);rows.append(r)
    print(c.get('native',False),c.get('heading'),r.get('elapsed_s'),r.get('joint_fraction'),flush=True)
    save('association',rows)
