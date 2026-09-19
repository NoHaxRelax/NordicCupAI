"""Fast full-game screening with frozen-source provenance.

Screening only: selected outcomes still require a Python-native recorded replay.
The policy receives only ordinary observations and simulation time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[2]
DEFAULT_FASTSIM = Path('/home/Ucals/projects/NordicCupAI-fastsim-inspect/survival-simulator')


def digest_tree(root):
    files = [*sorted((root/'models').rglob('*.py')), *sorted((root/'models').rglob('*.json'))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def atom(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                    help='Frozen simulator source containing models/ (and src/ for verification)')
    ap.add_argument('--fastsim', type=Path, default=DEFAULT_FASTSIM,
                    help='Source root containing the compiled fastsim package')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--seconds', type=float, default=3000.)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--verify-python', action='store_true',
                    help='Lockstep against Python; intended for a short bounded run')
    ap.add_argument('--check-every', type=int, default=1)
    args = ap.parse_args()
    source, fastroot = args.source.resolve(), args.fastsim.resolve()
    sys.path[:0] = [str(source), str(fastroot)]
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    out = args.output.resolve()
    provenance = dict(source=str(source), fastsim=str(fastroot), seed=args.seed,
                      requested_seconds=args.seconds, verify_python=args.verify_python,
                      policy_inputs='Unmodified per-agent DTO observations and simulation time only',
                      model_sha256=digest_tree(source),
                      engine_sha256=hashlib.sha256((fastroot/'fastsim/_engine.cpp').read_bytes()).hexdigest())
    for root, key in ((source, 'source_git'), (fastroot, 'fastsim_git')):
        try:
            import subprocess
            top = Path(subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'], text=True).strip())
            # A copied source directory inside another checkout is pinned by its
            # per-file hashes, not by the enclosing checkout's unrelated HEAD.
            provenance[key] = (subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
                               if top.resolve() in (root.resolve(), root.resolve().parent) else None)
        except Exception: provenance[key] = None
    atom(out/'manifest.json', provenance)
    started = time.monotonic()
    summary = dict(seed=args.seed, status='running', requested_seconds=args.seconds,
                   policy_inputs=provenance['policy_inputs'])
    atom(out/'summary.json', summary)
    try:
        # Import after path setup, so models comes from the requested frozen tree.
        from models.core import EntrapmentPolicy
        import fastsim
        policy = EntrapmentPolicy(seed=args.seed)
        fast = fastsim.SimulationCore(seed=args.seed)
        py = verifier = None
        if args.verify_python:
            # Importing verify installs creation-counter hashes before either
            # Python world is created. Native Python otherwise uses address-based
            # set iteration and is not seed-reproducible across processes.
            import fastsim.verify as verifier
            verifier._serial['c'] = __import__('itertools').count(1)
            py = verifier.PySim(seed=args.seed)
        actions, seen, peak, steps = [], set(), 0, 0
        policy_seconds = engine_seconds = python_seconds = 0.
        state = None
        while True:
            a = time.perf_counter(); sf = fast.step(actions); b = time.perf_counter()
            engine_seconds += b-a; steps += 1
            if py is not None:
                c=time.perf_counter(); sp=py.step(actions); d=time.perf_counter(); python_seconds += d-c
                diff=verifier.first_diff(sp,sf)
                if not diff and steps % args.check_every == 0:
                    diff=verifier.first_diff(verifier.world_py(py.env),verifier.world_fast(fast))
                    if not diff and py.env.rng.getstate()!=fast._engine.rng_state(): diff='rng state'
                if diff: raise RuntimeError(f'lockstep divergence at step {steps}: {diff}')
            state=sf; ids={s['agent_id'] for s in sf['observations']}; seen.update(ids); peak=max(peak,len(ids))
            if not ids or sf['sim_time'] >= args.seconds: break
            a=time.perf_counter(); actions=policy(sf['observations'],sf['sim_time']); b=time.perf_counter()
            policy_seconds += b-a
        metrics=dict(policy.metrics)
        summary.update(status='complete', sim_time=state['sim_time'], score=state['score'],
                       final_agents=state['num_agents'], peak_agents=peak,
                       total_agents_seen=len(seen), births=max(0,len(seen)-5), steps=steps,
                       wall_seconds=time.monotonic()-started,
                       policy_seconds=policy_seconds, engine_seconds=engine_seconds,
                       python_engine_seconds=python_seconds if py is not None else None,
                       policy_metrics=metrics, events=policy.events,
                       entrapment=dict(site_selected=policy.site is not None,
                           guide_assignments=metrics.get('guide_assignments'),
                           delivery_arrivals=metrics.get('delivery_arrivals'),
                           bait_arrivals=metrics.get('bait_arrivals'),
                           overlapping_replacements=metrics.get('overlapping_replacements')),
                       verification=('exact returned state, full world, and RNG lockstep; Python object hashes use creation counters'
                                     if py is not None else None))
    except Exception:
        summary.update(status='error', wall_seconds=time.monotonic()-started,
                       error=traceback.format_exc())
        atom(out/'summary.json', summary)
        raise
    atom(out/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('events','error')}, indent=2))


if __name__ == '__main__': main()
