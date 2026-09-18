"""Tick-by-tick trace of one test_leash case: python scripts/trapper/trace_leash.py --seed 2 --start 31 --end 35"""
import argparse, math, os, sys
from pathlib import Path
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts' / 'trapper'))
import test_leash as tl
from src.utils.DTOs import ActionRequest
from models.trapper.geometry import wrap, heading_of, sub, dist, add, mul
from models.trapper.lure import Lure, Delivery, predator_target
from models.trapper.motion import hold, action, step_toward, speed_for
from models.trapper.oracle import OracleWorld
from models.trapper.sites import find_gap_sites

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2); ap.add_argument('--start', type=float, default=0); ap.add_argument('--end', type=float, default=10)
ap.add_argument('--obstacles', type=int, default=6); ap.add_argument('--staffed', type=int, default=-1); ap.add_argument('--second', type=int, default=0)
a = ap.parse_args()
staffed = (a.seed % 2 == 0) if a.staffed < 0 else bool(a.staffed)
# replicate run_case setup
arena, rng = tl.build(a.seed, a.obstacles); env = arena.env
oracle = OracleWorld(env); world = oracle.update()
site = [s for s in find_gap_sites(world.rects, env.width, env.height) if s.normal[1] < 0 and abs(s.front_mid[0] - 800) < 1][0]
rects = world.rects
def free(p, r=8.0): return tl.free_point(p, r, rects, env.width, env.height)
for _ in range(500):
    g = (rng.uniform(80, 1520), rng.uniform(80, 760))
    if 350 <= dist(g, site.front_mid) <= 700 and free(g, 12): break
for _ in range(500):
    ang = rng.uniform(0, 2 * math.pi); r = rng.uniform(120, 220)
    pp = add(g, (r * math.cos(ang), r * math.sin(ang)))
    if free(pp, 14) and 30 < pp[0] < 1570 and 30 < pp[1] < 1170: break
guide = arena.add_agent(*g, heading=rng.uniform(0, 2 * math.pi), energy=500)
chasing = rng.random() < 0.6
heading = math.atan2(g[1] - pp[1], g[0] - pp[0]) if chasing else rng.uniform(0, 2 * math.pi)
pred = arena.add_predator(*pp, heading=heading, energy=rng.uniform(60, 190))
bait = arena.add_agent(*site.holder, heading=math.atan2(site.normal[1], site.normal[0]), energy=300) if staffed else None
if a.second:
    for _ in range(500):
        ang = rng.uniform(0, 2 * math.pi); r = rng.uniform(200, 320)
        p2 = add(g, (r * math.cos(ang), r * math.sin(ang)))
        if free(p2, 14) and 30 < p2[0] < 1570 and 30 < p2[1] < 1170: break
    arena.add_predator(*p2, heading=rng.uniform(0, 2 * math.pi), energy=rng.uniform(60, 190))
d = Delivery(site=site, guide=guide.agent_id, pid=0, become_bait=not staffed, created=0.0, leash=True)
state = arena.step([]); world = oracle.update(state['observations'])
print('mouth', site.front_mid, 'holder', tuple(round(v) for v in site.holder), 'far', site.far_mouth, 'guide start', tuple(round(v) for v in g), 'pred start', tuple(round(v) for v in pp), 'chasing', chasing)
for t in range(int(a.end * 10)):
    lure = Lure(world, held=set(), baits={bait.agent_id} if bait else set())
    holder = tl.Holder(world)
    actions = []
    for aid, ag in world.agents.items():
        if bait is not None and aid == bait.agent_id:
            act, _ = holder.act(ag, site.holder, site)
            if dist(ag.p, site.holder) < 0.8: act = action(aid, 0.0, 0.0, 0.0)
        elif aid == guide.agent_id:
            p = world.predator(0)
            if d.done is None:
                act = lure.act(d, ag, p) or hold(ag)
            elif d.phase == 'FLYBY' and d.flee_heading is not None:
                act = step_toward(ag, add(ag.p, mul(d.flee_heading, 40.0)), speed_for(ag, True))
            else:
                act = hold(ag)
        else:
            act = hold(ag)
        actions.append((aid, ActionRequest(**act)))
    state = arena.step(actions); world = oracle.update(state['observations'])
    p = world.predator(0); ga = world.agents.get(guide.agent_id)
    if state['sim_time'] >= a.start:
        tgt = predator_target(world, p)
        gap = dist(ga.p, p.p) if ga else -1
        bearing = math.degrees(abs(wrap(heading_of(sub(ga.p, p.p)) - p.heading))) if ga else -1
        print(f"t={state['sim_time']:5.1f} gap={gap:5.1f} pe={p.energy:5.1f} rest={int(bool(p.resting))} tgt={tgt} bear={bearing:4.0f} pspd={p.speed:4.1f} pred=({p.x:.0f},{p.y:.0f}) h={math.degrees(p.heading)%360:4.0f} g=({ga.x:.0f},{ga.y:.0f}) e={ga.energy:.0f} | {d.decision[:70]}" if ga else f"t={state['sim_time']:5.1f} guide dead")
    if ga is None and d.done != 'delivered': break
