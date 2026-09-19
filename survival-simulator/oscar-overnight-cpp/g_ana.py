import json, collections, sys, statistics as st
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
print(len(rows), 'rows; skips:', dict(collections.Counter(r.get('skip', 'ran') for r in rows)))
ran = [r for r in rows if 'skip' not in r]
if not ran: sys.exit()
print(f"overall delivered {sum(r['delivered'] for r in ran)}/{len(ran)} = {100*sum(r['delivered'] for r in ran)/len(ran):.0f}%; guide died {100*sum(1 for r in ran if not r['guide_alive'])/len(ran):.0f}%; bait died {sum(1 for r in ran if not r['bait_alive'])}")
for key in ('bear', 'dg', 'dp', 'speed'):
    vals = sorted({r[key] for r in ran})
    print(f"  by {key:5s}: " + '  '.join(f"{int(v)}: {100*sum(r['delivered'] for r in ran if r[key]==v)/max(1,sum(1 for r in ran if r[key]==v)):3.0f}% (n{sum(1 for r in ran if r[key]==v)})" for v in vals))
td = [r['t_deliver'] for r in ran if r['t_deliver']]
if td: print(f"  t_deliver median {st.median(td):.1f} s; guide energy used (survivors) n={sum(1 for r in ran if r['used'] is not None)}")
fails = collections.Counter()
for r in ran:
    if r['delivered']: continue
    s = r['states']
    fails['guide killed early' if r['t_guide_died'] and r['t_guide_died'] < 8 and not r['delivered'] else ('stuck in handoff' if s.endswith('3'*8) else ('lost/acquire loop' if '0' in s[2:] or s.endswith('1'*6) else 'other'))] += 1
print('  failure modes:', dict(fails))
