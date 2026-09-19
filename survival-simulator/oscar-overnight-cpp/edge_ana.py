"""Edge-case delivery scenarios (nightsim/guide.py with NIGHT_GE / NIGHT_NPRED / NIGHT_NBY): per label and setting,
delivery rate and a failure breakdown. usage: edge_ana.py rows.jsonl [more.jsonl]"""
import sys, json, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
ran = [r for r in rows if 'skip' not in r]
print(f"{len(rows)} rows, {len(ran)} valid; skips {dict(collections.Counter(r['skip'] for r in rows if 'skip' in r))}")
grp = collections.defaultdict(list)
for r in ran: grp[(r['label'], r.get('ge'), r.get('npred0', 1), r.get('nby', 0))].append(r)
for k in sorted(grp, key=str):
    rs = grp[k]; n = len(rs); d = sum(r['delivered'] for r in rs)
    fails = [r for r in rs if not r['delivered']]
    c = collections.Counter()
    for r in fails:
        gl = r.get('g_last')
        if not r['bait_alive']: c['bait died'] += 1
        elif r['t_guide_died'] is not None:
            capped = gl is not None and gl[0] < 160.
            slowb = gl is not None and len(gl) > 3 and gl[3] in (1, 2, 4) and gl[4] not in (1, 2, 4)
            c[('guide killed WALK-CAPPED' if capped else ('guide killed, guide on SLOW terrain, predator not' if slowb else 'guide killed with sprint energy, same terrain')) + (' <3 s' if r['t_guide_died'] < 3 else '')] += 1
        else: c['guide alive, predator not delivered'] += 1
    ok = [r for r in rs if r['delivered']]
    extra = ''
    if k[2] > 1: extra += f"  all {k[2]} predators held at the end: {sum(1 for r in rs if r['held_all'] >= k[2])}/{n}"
    if k[3] > 0: extra += f"  bystanders killed per scenario {sum(r['by_dead'] for r in rs)/n:.2f}"
    print(f"{k[0]:12s} GE {k[1]} pred {k[2]} by {k[3]}: delivered {d}/{n} = {100*d/n:.0f}%  guide survives {100*sum(r['guide_alive'] for r in rs)/n:.0f}% (on success {100*sum(r['guide_alive'] for r in ok)/max(1,len(ok)):.0f}%){extra}")
    print(f"     failures: {dict(c)}")
