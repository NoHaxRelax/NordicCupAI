"""Run a policy on unmodified upstream dynamics with read-only diagnostics.

Diagnostics wrap `Environment.kill_agent` and `remove_fruit` only to observe
which agent died and why (predator kill vs. starvation) and how much fruit was
eaten. They do not change engine state or RNG use. No online submissions.
"""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import sys, pathlib, argparse, time, json, platform, hashlib, importlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor' / 'survival-simulator'))
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(ROOT / 'research' / 'society'))
from src.core import SimulationCore

# SURVIVAL_ENGINE=fast runs the native port (survival/fastsim), which reproduces this
# engine step for step; see fastsim/README.md. harness_np sets PREDATORS = False.
ENGINE = os.environ.get('SURVIVAL_ENGINE', 'python')
PREDATORS = True


def load_policy(spec, seed, **kw):
    """spec: 'module:Class' or one of the legacy simple modes."""
    if spec in ('dummy', 'greedy', 'nursery', 'speed'):
        from simple_policies import SimplePolicy
        return SimplePolicy(spec, seed), 'simple_policies.py'
    mod, cls = spec.split(':')
    m = importlib.import_module(mod)
    return getattr(m, cls)(seed=seed, **kw), (pathlib.Path(m.__file__).name)


def run(spec, seed, horizon, out_dir, label='', record=False, verbose=False, sample_every=50., world_log=None, world_every=5., flee_log=None, **kw):
    start = time.perf_counter()
    fast = ENGINE == 'fast'
    if fast:
        sys.path.insert(0, str(ROOT))
        import fastsim
        if record:
            raise ValueError('--record needs the Python engine (the recorder reads engine objects)')
        sim = fastsim.SimulationCore(seed=seed, predators=PREDATORS)
    else:
        sim = SimulationCore(seed=seed)
    env = sim.env
    policy, pfile = load_policy(spec, seed, **kw)

    # --- read-only diagnostics ---
    diag = dict(deaths=[], fruit_eaten=0, fruit_energy=0., predator_energy_lost=0.)

    def take_events():
        # native engine: the same records the wrappers below collect
        for kind, t, aid, age, energy in sim.pop_events():
            if kind == 'fruit':
                diag['fruit_eaten'] += 1
                diag['fruit_energy'] += energy
            else:
                diag['deaths'].append(dict(t=round(t, 1), id=aid, age=round(age, 1), energy=round(energy, 2), cause=kind))
                if kind == 'predator':
                    diag['predator_energy_lost'] += energy
    phase = {'in_predator': False}
    if not fast:
        orig_kill, orig_remove_fruit, orig_non_agent_step = env.kill_agent, env.remove_fruit, env.non_agent_step

    def kill_agent(agent):
        cause = 'predator' if phase['in_predator'] else 'starvation'
        diag['deaths'].append(dict(t=round(env.time, 1), id=agent.agent_id, age=round(agent.age, 1),
                                   energy=round(agent.energy, 2), cause=cause))
        if cause == 'predator':
            diag['predator_energy_lost'] += agent.energy
        return orig_kill(agent)

    def remove_fruit(fruit):
        # Called both for eating and rotting; eating happens while an agent touches it.
        if phase.get('agent_scan') and any(((a.x-fruit.x)**2+(a.y-fruit.y)**2) ** .5 < a.size+fruit.radius for a in env.agents):
            diag['fruit_eaten'] += 1
            diag['fruit_energy'] += fruit.energy
        return orig_remove_fruit(fruit)

    # Predator kills happen inside non_agent_step after the agent loop. We detect the
    # phase by watching which loop is running: agent starvation kills occur while
    # `agent_scan` is set (first loop), predator kills afterwards.
    class _PredList(list):
        pass

    def non_agent_step(dt):
        phase['agent_scan'] = True
        phase['in_predator'] = False
        # The agent loop runs first; we flip flags when the predator loop begins by
        # observing the first access to env.predators via a property-less trick:
        # wrap predators list iteration.
        preds = env.predators
        class Watch(list):
            def __iter__(inner):
                phase['agent_scan'] = False
                phase['in_predator'] = True
                return list.__iter__(inner)
        env.predators = Watch(preds)
        try:
            r = orig_non_agent_step(dt)
        finally:
            new = list(env.predators)
            env.predators = preds
            preds[:] = new
            phase['in_predator'] = False
            phase['agent_scan'] = False
        return r

    if not fast:
        env.kill_agent, env.remove_fruit, env.non_agent_step = kill_agent, remove_fruit, non_agent_step

    recorder = None
    if record:
        from recorder import ReplayRecorder
        recorder = ReplayRecorder(env, title=f'{spec} seed {seed} {label}', policy=spec, seed=seed,
                                  scenario='generated', native_render=False, every=50,
                                  notes='state-only diagnostic sweep; generated map; observation-only policy')

    state = sim.step([])  # same first empty step as the official server
    if fast: take_events()
    peak = state['num_agents']; samples = []; next_sample = sample_every
    if world_log: pathlib.Path(world_log).parent.mkdir(parents=True, exist_ok=True)
    wlog = open(world_log, 'w') if world_log else None; next_world = 0.
    if flee_log: pathlib.Path(flee_log).parent.mkdir(parents=True, exist_ok=True)
    flog = open(flee_log, 'w') if flee_log else None
    import math as _m
    def dump_world():
        wlog.write(f"== t={state['sim_time']:.1f} alive={len(env.agents)} preds={len(env.predators)} trees={len(env.trees)} fruits={len(env.fruits)} score={state['score']:.1f}\n")
        for a in sorted(env.agents, key=lambda a: a.agent_id):
            nf = sum(1 for f in env.fruits if _m.dist((a.x, a.y), (f.x, f.y)) < 100)
            nt = sum(1 for t in env.trees if _m.dist((a.x, a.y), (t.x, t.y)) < 100)
            pd = min((_m.dist((a.x, a.y), (p.x, p.y)) for p in env.predators), default=None)
            wlog.write(f"  a{a.agent_id} pos=({a.x:.0f},{a.y:.0f}) age={a.age:.0f} e={a.energy:.0f} maxE={a.max_energy:.0f} spd={a.speed:.0f}/{a.sprint_speed:.0f} hear={a.hearing_radius:.0f} vis={a.vision_radius:.0f}/{a.cone_angle:.2f} mode={getattr(policy, 'decisions', {}).get(a.agent_id)} fr100={nf} tr100={nt} predDist={pd and round(pd)}\n")
        for i, p in enumerate(env.predators):
            wlog.write(f"  P{i} pos=({p.x:.0f},{p.y:.0f}) rest={p.resting} e={p.energy:.0f}\n")
        wlog.flush()
    while state['num_agents'] and state['sim_time'] < horizon:
        actions = policy(state['observations'], state['sim_time'])
        if flog is not None and env.predators:
            obs_by = {o['agent_id']: o for o in state['observations']}
            for aid, act in actions:
                ag = env.agents_dict.get(aid)
                if ag is None: continue
                pd, pr = min(((_m.dist((ag.x, ag.y), (p.x, p.y)), p) for p in env.predators), key=lambda x: x[0])
                if pd > 140: continue
                po = [(round(o['distance']), round(o['angle'], 2), round(o['rel_dir'], 2)) for o in obs_by[aid]['observations'] if o['type'] == 'Predator']
                dec = getattr(policy, 'decisions', {}).get(aid)
                flog.write(f"t={state['sim_time']:.1f} a{aid} e={ag.energy:.0f} age={ag.age:.0f} pos=({ag.x:.0f},{ag.y:.0f}) dir={ag.direction%6.2832:.2f} biome={obs_by[aid]['biome']} hear={ag.hearing_radius:.0f} vis={ag.vision_radius:.0f}/{ag.cone_angle:.2f} | P d={pd:.0f} pos=({pr.x:.0f},{pr.y:.0f}) dir={pr.direction%6.2832:.2f} rest={pr.resting} pe={pr.energy:.0f} | obs={po} | dec={dec} act=(d={act.move_distance:.1f},dir={act.move_direction:.2f},turn={act.turn_angle:.2f})\n")
        state = sim.step(actions)
        if fast: take_events()
        if recorder: recorder.capture(actions)
        if wlog and state['sim_time'] >= next_world-1e-6:
            next_world += world_every; dump_world()
        peak = max(peak, state['num_agents'])
        if state['sim_time'] >= next_sample - 1e-6:
            next_sample += sample_every
            ages = [a.age for a in env.agents]
            samples.append(dict(time=round(state['sim_time'], 1), score=round(state['score'], 3),
                                alive=state['num_agents'], predators=len(env.predators),
                                trees=len(env.trees), fruits=len(env.fruits),
                                mean_energy=round(sum(a.energy for a in env.agents)/max(1, len(env.agents)), 1),
                                max_age=round(max(ages), 1) if ages else 0,
                                fruit_eaten=diag['fruit_eaten']))
            if verbose:
                print(json.dumps(samples[-1]), flush=True)
    pred_deaths = [d for d in diag['deaths'] if d['cause'] == 'predator']
    result = dict(policy=spec, label=label, seed=seed, horizon=horizon,
                  survival_seconds=round(state['sim_time'], 2), score=round(state['score'], 4),
                  alive=state['num_agents'], peak_agents=peak, total_agents_created=env._next_agent_id,
                  predators=len(env.predators), fruit_eaten=diag['fruit_eaten'],
                  fruit_score=round(diag['fruit_energy']/1000, 4),
                  predator_deaths=len(pred_deaths), starvation_deaths=len(diag['deaths'])-len(pred_deaths),
                  predator_penalty=round(diag['predator_energy_lost']/100, 4),
                  mean_energy_at_predator_death=round(sum(d['energy'] for d in pred_deaths)/max(1, len(pred_deaths)), 1),
                  deaths=diag['deaths'], samples=samples,
                  policy_metrics=getattr(policy, 'metrics', None),
                  wall_seconds=round(time.perf_counter()-start, 1), engine=ENGINE,
                  platform=platform.platform(), python=platform.python_version(),
                  policy_sha256=hashlib.sha256((ROOT/'research'/'society'/pfile).read_bytes()).hexdigest() if (ROOT/'research'/'society'/pfile).exists() else None,
                  source_commit='acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
                  limitations='Local Linux generated maps; no hidden state used by the policy; diagnostics read engine state only.')
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{spec.replace(':','-')}{('-'+label) if label else ''}-seed{seed}"
    (out / f'{name}.json').write_text(json.dumps(result, indent=1) + '\n')
    if wlog: wlog.close()
    if flog: flog.close()
    if recorder:
        (out / 'replays').mkdir(exist_ok=True)
        recorder.save(out / 'replays' / f'{name}-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json.gz')
    brief = {k: v for k, v in result.items() if k not in ('samples', 'deaths', 'limitations', 'platform', 'python', 'policy_sha256', 'source_commit', 'policy_metrics')}
    print(json.dumps(brief), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='society:SocietyPolicy')
    p.add_argument('--seeds', type=int, nargs='+', default=[1])
    p.add_argument('--horizon', type=float, default=3000)
    p.add_argument('--out', default=str(ROOT / 'results' / 'society'))
    p.add_argument('--label', default='')
    p.add_argument('--record', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--kw', default='{}', help='JSON kwargs for the policy constructor')
    p.add_argument('--world-log', default=None, help='write true world state (agents, predators, counts) every --world-every seconds to this path (one file per seed, seed appended)')
    p.add_argument('--world-every', type=float, default=5.)
    p.add_argument('--flee-log', default=None, help='per-tick trace of every agent within 140 of a predator (same run)')
    a = p.parse_args()
    for seed in a.seeds:
        wl = None if a.world_log is None else (a.world_log if len(a.seeds) == 1 else f'{a.world_log}.seed{seed}')
        fl = None if a.flee_log is None else (a.flee_log if len(a.seeds) == 1 else f'{a.flee_log}.seed{seed}')
        run(a.policy, seed, a.horizon, a.out, a.label, a.record, a.verbose, world_log=wl, world_every=a.world_every, flee_log=fl, **json.loads(a.kw))
