"""Matched founder controls on the exact arranged training resources."""
from experiment import run,save,compact
import json
rows=[]
for food in ('none','line','orchard'):
    for policy in ('adaptive','nofood','predictive'):
        r=run(dict(energy=150,food=food),policy,140)
        rows.append(r);print(json.dumps(compact(r)),flush=True)
save('controls-final',rows)
