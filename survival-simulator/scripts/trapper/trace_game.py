"""Trace every delivery of a full trapper game tick by tick (throttled).

    ../.venv/bin/python scripts/trapper/trace_game.py --seed 2 --seconds 600 --every 5
"""
import argparse, os, sys, math
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.core import SimulationCore
from models.trapper.policy import TrapperPolicy
from models.trapper.lure import predator_target
from models.trapper.geometry import dist

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2); ap.add_argument('--seconds', type=float, default=600)
ap.add_argument('--every', type=int, default=5); ap.add_argument('--max-lines', type=int, default=400)
a = ap.parse_args()
sim = SimulationCore(seed=a.seed); env = sim.env
pol = TrapperPolicy(seed=a.seed, env=env)
state = sim.step([]); tick = 0; lines = 0; seen_events = 0
while state['num_agents'] and state['sim_time'] < a.seconds and lines < a.max_lines:
    acts = pol(state['observations'], state['sim_time']); state = sim.step(acts); tick += 1
    m = pol.manager; w = pol.world
    for e in m.events[seen_events:]:
        if e['kind'] not in ('senescent',):
            print('EVENT', e); lines += 1
    seen_events = len(m.events)
    if tick % a.every == 0 and m.deliveries and w is not None:
        for pid, d in m.deliveries.items():
            g = w.agents.get(d.guide); p = w.predator(pid)
            if g is None or p is None: continue
            tgt = predator_target(w, p)
            others = sorted((dist(q.p, g.p), q.pid) for q in w.predators if q.pid != pid)[:1]
            print(f"t={w.time:6.1f} pid={pid} guide={d.guide} e={g.energy:5.0f} age={g.age:4.0f} {d.phase:8s} gap={dist(g.p, p.p):5.0f} "
                  f"ptarget={tgt} pspd={p.speed:4.1f} rest={p.resting} site_d={dist(g.p, d.site.corridor_start):4.0f} near={others} | {d.decision}")
            lines += 1
