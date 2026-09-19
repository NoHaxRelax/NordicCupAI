import json, collections, sys
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
R = {}
for r in rows:
    if r.get('skip'): continue
    R[(r['label'], r['seed'], r['speed'], r['D'], r['B'])] = r
rows = list(R.values())
labs = sorted({r['label'] for r in rows})
print('kill % by agent walking speed (rows) and start distance D; cells = predator in front/side/behind')
for lab in labs:
    agg = collections.defaultdict(list)
    for r in rows:
        if r['label'] == lab: agg[(r['speed'], r['D'], r['B'])].append(r['killed'])
    tot = [r['killed'] for r in rows if r['label'] == lab]
    print(f'{lab:<9} overall {100*sum(tot)/len(tot):4.1f}%  ' + ' | '.join(f"sp{int(sp)}: " + ' '.join('/'.join(f"{100*sum(agg[(sp,D,B)])/max(1,len(agg[(sp,D,B)])):.0f}" for B in (0,90,180)) for D in (40,80,150,250)) for sp in (10,15,20)))
