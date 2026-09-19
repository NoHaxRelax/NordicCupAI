"""Paired analysis: python pana.py rows.jsonl [ref_label]"""
import json, sys, math, collections
rows = [json.loads(l) for f in sys.argv[1].split(',') for l in open(f) if l.strip()]
ref = sys.argv[2] if len(sys.argv) > 2 else 'ref'
by = collections.defaultdict(dict)
for r in rows: by[r['label']][r['seed']] = r
def ms(v): 
    n = len(v); m = sum(v)/n; sd = math.sqrt(sum((x-m)**2 for x in v)/max(1, n-1)); return m, sd/math.sqrt(n)
print(f"{'label':16s} {'n':>4s} {'surv':>7s} {'score':>7s} {'dSurv':>12s} {'dScore':>12s} {'kills':>6s} {'full':>5s} tp(n,ins,last,kids,free_t,pris)")
for lab in sorted(by, key=lambda l: (l != ref, l)):
    d = by[lab]; common = [s for s in d if s in by[ref]]
    sv = [d[s]['surv'] for s in d]; sc = [d[s]['score'] for s in d]
    ds = ms([d[s]['surv'] - by[ref][s]['surv'] for s in common]) if common and lab != ref else (0, 0)
    dc = ms([d[s]['score'] - by[ref][s]['score'] for s in common]) if common and lab != ref else (0, 0)
    tp = [d[s].get('tp') for s in d if d[s].get('tp')]
    tpm = [round(sum(t[i] for t in tp)/len(tp), 1) for i in range(6)] if tp else ''
    print(f"{lab:16s} {len(d):4d} {sum(sv)/len(sv):7.0f} {sum(sc)/len(sc):7.0f} {ds[0]:+6.0f}±{ds[1]:<4.0f} {dc[0]:+6.0f}±{dc[1]:<4.0f} {sum(d[s]['pdeaths'] for s in d)/len(d):6.1f} {sum(1 for s in d if d[s]['surv']>=2999)/len(d):5.2f} {tpm}")
