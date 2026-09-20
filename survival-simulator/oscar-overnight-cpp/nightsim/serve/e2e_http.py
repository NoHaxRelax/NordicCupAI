"""Unmodified Python engine (/opt/nordiccup/src) <-> HTTP endpoint, payloads through the organiser DTOs (pydantic JSON)."""
import sys, time, json, math, urllib.request
sys.path.insert(0, '/opt/nordiccup')
from src.core import SimulationCore
from DTOs import ActionRequest, StepResponse
url, seed, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
sim = SimulationCore(seed=seed); env = sim.env
state = sim.step([]); t0 = time.time(); n = 0; nulls = 0; omitted = 0
while state['observations'] and state['sim_time'] < T:
    body = StepResponse(game_status='running', score=state['score'], sim_time=state['sim_time'], n_agents=state['num_agents'],
                        agent_status=state['observations']).model_dump_json()
    nulls += body.count('null')
    req = urllib.request.Request(url, data=body.encode(), headers={'Content-Type': 'application/json'})
    acts = json.loads(urllib.request.urlopen(req, timeout=30).read())['actions']
    omitted += len(state['observations']) - len(acts)
    state = sim.step([(a['agent_id'], ActionRequest(**a)) for a in acts]); n += 1
    if n % 2000 == 0:
        fr = sum(1 for p in env.predators if p.x == 1590. and p.y == 1190.)
        print(f"t={state['sim_time']:.0f} agents={state['num_agents']} preds={len(env.predators)} frozen={fr} score={state['score']:.1f} nulls={nulls} omitted={omitted} wall={time.time()-t0:.0f}s", flush=True)
fr = sum(1 for p in env.predators if p.x == 1590. and p.y == 1190.)
print(f"END t={state['sim_time']:.1f} score={state['score']:.1f} agents={state['num_agents']} preds={len(env.predators)} frozen={fr} nulls={nulls} omitted={omitted} wall={time.time()-t0:.0f}s", flush=True)
