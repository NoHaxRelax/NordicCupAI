"""Hyperparameter search for the orchard policy: random round, then hill-climbing rounds around the
best configurations. Dependency-free (no optuna); each round is one sweep on a process pool.

Usage (one shard per pod; shards draw disjoint samples):
  opt.py --space space.json --round 1 --samples 8 --shard 0/4 --seeds 1 2 3 4 5 6 7 8 --workers 32 --out DIR
  opt.py --space space.json --round 2 --samples 8 --shard 0/4 --seeds ... --out DIR --prior journal1.jsonl journal2.jsonl ...
Round 1 samples uniformly from the space. Round >=2 takes the top-k configs from the prior journals and
perturbs them (each parameter with probability 0.4, by +-25% of its range or a category switch).
Objective per config: mean score (survival seconds + fruit score) over the seeds. Every run's brief
result is appended to <out>/journal.jsonl; the sweep's per-run JSONs are kept in <out>/runs.
"""
import os, sys, json, argparse, pathlib, random, time, statistics
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research'/'orchard'))


def sample_space(space, rng):
    cfg = {}
    for k, spec in space.items():
        if isinstance(spec, dict) and 'choices' in spec: cfg[k] = rng.choice(spec['choices'])
        elif isinstance(spec, dict) and spec.get('type') == 'int': cfg[k] = rng.randint(spec['low'], spec['high'])
        elif isinstance(spec, dict) and spec.get('log'): cfg[k] = round(10**rng.uniform(__import__('math').log10(spec['low']), __import__('math').log10(spec['high'])), 4)
        else: cfg[k] = round(rng.uniform(spec['low'], spec['high']), 4)
    return cfg


def midpoint(spec, rng):
    if isinstance(spec, dict) and 'choices' in spec: return rng.choice(spec['choices'])
    if spec.get('type') == 'int': return (spec['low']+spec['high'])//2
    return round((spec['low']+spec['high'])/2, 4)


def perturb(cfg, space, rng, p=0.4):
    new = {k: cfg.get(k, midpoint(spec, rng)) for k, spec in space.items()}   # tolerate parameters added later
    cfg = dict(new)
    for k, spec in space.items():
        if rng.random() > p: continue
        if isinstance(spec, dict) and 'choices' in spec: new[k] = rng.choice(spec['choices'])
        elif spec.get('type') == 'int':
            span = max(1, (spec['high']-spec['low'])//4); new[k] = int(min(spec['high'], max(spec['low'], cfg[k]+rng.randint(-span, span))))
        else:
            span = (spec['high']-spec['low'])*0.25; new[k] = round(min(spec['high'], max(spec['low'], cfg[k]+rng.uniform(-span, span))), 4)
    return new


def load_prior(paths):
    by = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line); key = json.dumps(r['cfg'], sort_keys=True)
            by.setdefault(key, {'cfg': r['cfg'], 'scores': [], 'surv': []})
            by[key]['scores'].append(r.get('eff_score', r['score'])); by[key]['surv'].append(r.get('eff_survival', r['survival']))
    rows = [dict(cfg=v['cfg'], n=len(v['scores']), score=statistics.mean(v['scores']), survival=statistics.mean(v['surv'])) for v in by.values()]
    rows.sort(key=lambda r: -r['score'])
    return rows


def _run(job):
    label, kw, seed, horizon, out, policy = job
    import harness_np, harness  # noqa
    r = harness.run(policy, seed, horizon, out, label, False, False, sample_every=100., **kw)
    T = r['survival_seconds']; last = [x for x in r['samples'] if x['time'] <= T+1e-6]
    trees_d = last[-1]['trees'] if last else None; fruits_d = last[-1]['fruits'] if last else None
    # tree-wall credit: a colony that died with <=2 trees and <=5 fruit on the map could not have gone on
    wall_death = T < horizon-1e-6 and trees_d is not None and trees_d <= 2 and fruits_d <= 5
    eff = horizon if wall_death else T
    return dict(label=label, cfg=kw, seed=seed, survival=T, score=r['score'], fruit=r['fruit_score'],
                eaten=r['fruit_eaten'], peak=r['peak_agents'], wall=r['wall_seconds'],
                trees_death=trees_d, fruits_death=fruits_d, wall_death=wall_death, eff_survival=eff,
                eff_score=eff+r['fruit_score'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--space', required=True); ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--samples', type=int, default=8); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument('--horizon', type=float, default=3000); ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', required=True); ap.add_argument('--prior', nargs='*', default=[])
    ap.add_argument('--topk', type=int, default=6); ap.add_argument('--policy', default='orchard:OrchardPolicy')
    ap.add_argument('--rng', type=int, default=0)
    ap.add_argument('--halving', type=int, default=0, help='successive halving: run this many seeds first, the rest only for configs within --slack of the prior leader')
    ap.add_argument('--slack', type=float, default=250.)
    a = ap.parse_args()
    space = json.load(open(a.space)); shard, nshard = map(int, a.shard.split('/'))
    rng = random.Random(1000*a.round+a.rng)
    if a.round == 1 or not a.prior:
        cfgs = [sample_space(space, rng) for _ in range(a.samples*nshard)][shard::nshard]
    else:
        prior = load_prior(a.prior)[:a.topk]
        cfgs = []
        for i in range(a.samples*nshard):
            cfgs.append(perturb(prior[i % len(prior)]['cfg'], space, rng))
        cfgs = cfgs[shard::nshard]
    out = pathlib.Path(a.out); (out/'runs').mkdir(parents=True, exist_ok=True)
    labels = {i: f"r{a.round}s{shard}c{i}" for i in range(len(cfgs))}
    from multiprocessing import Pool
    t0 = time.perf_counter(); rows = []
    def run_jobs(pool, journal, jobs):
        for r in pool.imap_unordered(_run, jobs):
            rows.append(r); journal.write(json.dumps(r)+'\n'); journal.flush()
            print(json.dumps({k: v for k, v in r.items() if k != 'cfg'}), flush=True)
    with Pool(a.workers) as pool, open(out/'journal.jsonl', 'a') as journal:
        if a.halving and a.halving < len(a.seeds):
            first, rest = a.seeds[:a.halving], a.seeds[a.halving:]
            jobs = [(labels[i], cfg, seed, a.horizon, str(out/'runs'), a.policy) for i, cfg in enumerate(cfgs) for seed in first]
            print(f"round {a.round} shard {shard}/{nshard}: stage 1, {len(cfgs)} configs x {len(first)} seeds = {len(jobs)} runs", flush=True)
            run_jobs(pool, journal, jobs)
            leader = load_prior(a.prior)[0]['score'] if a.prior else max(statistics.mean(r['eff_score'] for r in rows if r['label'] == l) for l in labels.values())
            keep = [i for i, l in labels.items() if statistics.mean(r['eff_score'] for r in rows if r['label'] == l) >= leader-a.slack]
            jobs = [(labels[i], cfgs[i], seed, a.horizon, str(out/'runs'), a.policy) for i in keep for seed in rest]
            print(f"stage 2: {len(keep)} of {len(cfgs)} configs within {a.slack:.0f} of leader {leader:.0f} get {len(rest)} more seeds = {len(jobs)} runs", flush=True)
            run_jobs(pool, journal, jobs)
        else:
            jobs = [(labels[i], cfg, seed, a.horizon, str(out/'runs'), a.policy) for i, cfg in enumerate(cfgs) for seed in a.seeds]
            print(f"round {a.round} shard {shard}/{nshard}: {len(cfgs)} configs x {len(a.seeds)} seeds = {len(jobs)} runs", flush=True)
            run_jobs(pool, journal, jobs)
    by = {}
    for r in rows: by.setdefault(r['label'], []).append(r)
    print(f"\n{len(rows)} runs in {time.perf_counter()-t0:.0f}s")
    for label, rs in sorted(by.items(), key=lambda x: -statistics.mean(r['eff_score'] for r in x[1])):
        print(f"{label:<10} eff_score {statistics.mean(r['eff_score'] for r in rs):7.1f} surv {statistics.mean(r['survival'] for r in rs):6.0f} walls {sum(1 for r in rs if r['wall_death'])} fruit {statistics.mean(r['fruit'] for r in rs):5.1f} full {sum(1 for r in rs if r['survival']>=a.horizon-1e-6)}/{len(rs)} cfg {json.dumps(rs[0]['cfg'])}")
