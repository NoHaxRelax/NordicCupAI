import json, sys
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
print(len(rows), 'maps; with >=1 site:', sum(1 for r in rows if r['n_sites'] > 0), ' mean confirmed walls', round(sum((r['walls'] or 0) for r in rows)/len(rows)))
tops = [r['graded'][0] for r in rows if r['graded']]
allg = [g for r in rows for g in r['graded']]
for name, gs in (('top site', tops), ('all graded', allg)):
    ok = [g for g in gs if g['free'] and g['blocked'] and g['lane']]
    held = [g for g in ok if 'hold_killed' in g]
    print(f"{name}: {len(gs)} graded, free {sum(g['free'] for g in gs)}, pred-blocked {sum(g['blocked'] for g in gs)}, lane {sum(g['lane'] for g in gs)}, all-valid {len(ok)}; "
          f"holds {len(held)}: bait survived {sum(1 for g in held if not g['hold_killed'])}, pred closest mean {sum(g['hold_dmin'] for g in held)/max(1,len(held)):.1f}")
print('maps with a valid site among top-3:', sum(1 for r in rows if any(g['free'] and g['blocked'] and g['lane'] for g in r['graded'])))
