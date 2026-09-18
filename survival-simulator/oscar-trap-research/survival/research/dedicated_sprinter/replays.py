"""Representative success and failure replays through the existing recorder."""
from experiment import run,save,compact,OUT
from protection import protect
import json

def main():
    cases=[('final-founder-orchard',dict(energy=150,food='orchard')),
           ('final-founder-no-food',dict(energy=150,food='none')),
           ('final-specialist-orchard',dict(energy=150,walk=12,capacity=300,food='orchard')),
           ('final-newborn-cutoff',dict(energy=75,food='orchard')),
           ('final-real-mutant',dict(energy=75,walk=13.934350918478602,sprint=20,capacity=309.4519473923729,food='orchard'))]
    heldout=json.loads((OUT/'validation-final-v5.json').read_text())['runs']
    failure=next(r['scenario'] for r in heldout if r['policy']=='predictive' and r['scenario']['seed']==407)
    cases.append(('final-heldout-capture',failure))
    rows=[]
    for name,c in cases:
        r=run(c,seconds=140,record=name);rows.append(r);print(json.dumps(compact(r)),flush=True)
    save('replays-final',rows)
    c=dict(seed=306,heading=-.3,position=[1100,700],rotation=2.3,energy=150,food='orchard')
    r=protect(c,'predictive',record='final-protected-workers');save('protection-replay',[r])
if __name__=='__main__':main()
