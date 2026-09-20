"""EXP12: energy cost of covering D px in ONE tick.
   1 sprint action  : speed*0.05 + (D-speed)*0.5           (D capped at sprint_speed)
   k walk actions   : D*0.05                               (no cap on total D)"""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\nikol\AppData\Local\Temp\claude\c--Users-nikol-OneDrive-Dokumenter-GitHub-NordicCupAI-sim-optimization\e503679d-d2bc-4239-9072-849f21af62bd\scratchpad\objfn")
from harness import *
from src.elements.environment import Environment
Environment.spawn_predator = lambda self,*a,**k: None

def solo(seed=71):
    env=build(seed); a=env.agents[0]
    for z in list(env.agents):
        if z is not a: env.kill_agent(z)
    env.agents_dict={a.agent_id:a}
    for _ in range(900):
        px=env.rng.uniform(500,1100); py=env.rng.uniform(400,800)
        if env._is_position_free(px-120,py-120,240,240): break
    a.x,a.y=px,py; a.energy=500.0; a.direction=0.0; env._update_agent_grid()
    return env,a

print("agent speed=10 sprint_speed=20 (founder). Cost measured as energy spent in the action phase.")
print("%6s | %-28s | %-28s | %s"%("D px","1 sprint action","ceil(D/10) walk actions","saving"))
for D in (10,15,20,40,80,200):
    env,a=solo(); e0=a.energy; x0=a.x
    d=min(D,a.sprint_speed)
    env.agent_step(a.agent_id, move_distance=float(d), move_direction=0.0, turn_angle=0.0)
    c_sprint=e0-a.energy; moved_s=abs(a.x-x0)
    env,b=solo(); e0=b.energy; x0=b.x
    k=int(np.ceil(D/10.0))
    for _ in range(k):
        env.agent_step(b.agent_id, move_distance=10.0, move_direction=0.0, turn_angle=0.0)
    c_walk=e0-b.energy; moved_w=abs(b.x-x0)
    print("%6d | cost %6.2f, moved %6.1f px  | cost %6.2f, moved %6.1f px (k=%2d) | %5.1fx cheaper per px"
          % (D, c_sprint, moved_s, c_walk, moved_w, k,
             (c_sprint/max(moved_s,1e-9))/(c_walk/max(moved_w,1e-9))))
print("\nderived: duplicate WALK actions cost a flat 0.05 energy/px at ANY per-tick distance;")
print("         a single action above `speed` costs 0.5 energy/px for the excess -> 10x.")
