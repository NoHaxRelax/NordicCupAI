"""One prepared native-map continuous-bait handoff, fully recorded."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, sys
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"research"),str(ROOT/"debugger"),str(ROOT/"vendor/survival-simulator")]
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest, ObservationResponse
from streaming_recorder_v2 import ReplayRecorder
from replaceable_sites.handoff_policy import StaticHandoffPolicy

OUT=ROOT/"results"/"replaceable_sites"

def state(env):
    return [ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]

def main(seed=10000, seconds=60., boundary=False, expanded_narrowest=False):
    env=SimulationCore(starting_agents=0,starting_predators=0,seed=seed).env
    static={"width":env.width,"height":env.height,"obstacles":[
        {"x":o.x,"y":o.y,"width":o.width,"height":o.height} for o in env.obstacles]}
    kind="expanded_narrowest" if expanded_narrowest else "boundary" if boundary else "interior"
    policy=StaticHandoffPolicy(json.loads(json.dumps(static)),site_kind=kind)
    site=policy.site
    # This seed's selected site is an interior pair; keep this proof focused on
    # continuous replacement rather than the separate boundary-lane extension.
    if not expanded_narrowest: assert bool(site["boundary_indices"]) == boundary
    inward=tuple(site["inward"]); mouth=tuple(site["mouth"])
    old=Agent(*site["goal"],energy=500,max_age=10_000,rng=env.rng); old.agent_id=0
    new=Agent(*site["replacement_entry"],energy=500,max_age=10_000,rng=env.rng); new.agent_id=1
    heading=math.atan2(inward[1],inward[0]); old.direction=heading; new.direction=heading+math.pi
    cross=tuple(site["cross"])
    if boundary or site["offset_approach"]:
        predstart=(mouth[0]-inward[0]*11.01+cross[0]*site["approach_lane_offset"],
                   mouth[1]-inward[1]*11.01+cross[1]*site["approach_lane_offset"])
    else:
        predstart=(mouth[0]-inward[0]*60,mouth[1]-inward[1]*60)
    predator=Predator(*predstart,energy=200,rng=env.rng)
    predator.direction=math.atan2(site["goal"][1]-predstart[1],site["goal"][0]-predstart[0]); predator.resting=False
    env.agents=[old,new];env.agents_dict={0:old,1:new};env._next_agent_id=2
    env.predators=[predator];env._update_spatial_grid()
    label="expanded-narrow-" if expanded_narrowest else "boundary-" if boundary else ""
    tag=f"native-map-{seed}-{label}continuous-replacement-{uuid4().hex[:8]}"
    replay=OUT/"replays"/f"{tag}.json.gz"
    rec=ReplayRecorder(env,title=f"Prepared continuous bait replacement · native map {seed}",
        policy="static-map DTO-only opposite-mouth boundary handoff v1" if boundary else "static-map DTO-only opposite-mouth handoff v1",seed=seed,every=1,
        native_render=True,native_width=400,
        scenario="exact native generated map; prepared interior refuge, bait, replacement, and one approaching predator",
        notes=("Every 0.1s frame retained. Infinite agent energy. Policy receives immutable static map, role IDs, "
               "native observation DTOs, and public time only. Fixture positions and predator truth are evaluator-only. "
               "This is a prepared retention/handoff proof, not autonomous deployment or general reliability."))
    rec.capture(); states=state(env); trace=[]; deaths=[]
    old_exit=False; replacement_goal=False; contained=[]
    for tick in range(round(seconds*10)):
        for a in env.agents:a.energy=a.max_energy
        inputs=json.loads(json.dumps(states)); actions=policy.act(inputs,env.time); pairs=[]; before=list(env.agents)
        assert set(a["agent_id"] for a in actions)==set(env.agents_dict)
        for action in actions:
            req=ActionRequest(**action); pairs.append((req.agent_id,req));env.agent_step(**action)
        env.non_agent_step(.1)
        deaths += [{"id":a.agent_id,"time":round(env.time,1)} for a in before if a.agent_id not in env.agents_dict]
        new_alive=1 in env.agents_dict; old_alive=0 in env.agents_dict
        replacement_goal |= new_alive and math.dist((env.agents_dict[1].x,env.agents_dict[1].y),site["goal"])<3
        old_exit |= old_alive and math.dist((env.agents_dict[0].x,env.agents_dict[0].y),site["replacement_entry"])<5
        along=(predator.x-mouth[0])*inward[0]+(predator.y-mouth[1])*inward[1]
        held=math.dist((predator.x,predator.y),mouth)<75 and along<=10
        contained.append(held)
        if tick%10==0:trace.append({"time":round(env.time,1),"old_alive":old_alive,"replacement_alive":new_alive,
            "replacement_at_goal":bool(replacement_goal),"old_exited_rear":bool(old_exit),"predator_held":bool(held)})
        rec.capture(pairs,policy.decisions,inputs=states,action_t=env.time-.1);states=state(env)
    rec.save(replay,reason="60-second prepared handoff horizon complete")
    result={"schema":"replaceable-site-native-handoff-v1","map_seed":seed,"seconds":round(env.time,1),
      "site":site,"boundary_fixture":bool(site["boundary_indices"]),"expanded_narrowest_fixture":expanded_narrowest,"old_exited_opposite_mouth":bool(old_exit),"replacement_reached_goal":bool(replacement_goal),
      "replacement_alive":bool(1 in env.agents_dict),"old_bait_alive":bool(0 in env.agents_dict),
      "deaths":deaths,"predator_held_final_30s":bool(all(contained[-300:])),"trace":trace,
      "replay":str(replay.relative_to(ROOT)),"frames_expected":round(seconds*10)+1,
      "policy_sha256":hashlib.sha256((HERE/"handoff_policy.py").read_bytes()).hexdigest(),
      "requested_seconds":seconds,"reason":"horizon","success":bool(replacement_goal and old_exit and 1 in env.agents_dict and all(contained[-300:])),
      "setup_label":"prepared boundary refuge continuous handoff" if boundary else "prepared interior refuge continuous handoff",
      "limitations":"Prepared fixture; geometric opposite-mouth access and one continuous handoff only."}
    OUT.mkdir(parents=True,exist_ok=True);receipt=OUT/f"{tag}.json";receipt.write_text(json.dumps(result,indent=2)+"\n")
    with gzip.open(replay,"rt") as f:data=json.load(f)
    assert len(data["frames"])==result["frames_expected"] and all("native_image" in f for f in data["frames"])
    print(json.dumps({k:result[k] for k in ("replacement_reached_goal","old_exited_opposite_mouth","replacement_alive","old_bait_alive","predator_held_final_30s","replay")},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--boundary",action="store_true")
    parser.add_argument("--expanded-narrowest",action="store_true");parser.add_argument("--seed",type=int,default=10000)
    parser.add_argument("--seconds",type=float,default=60.);args=parser.parse_args()
    main(seed=args.seed,seconds=args.seconds,boundary=args.boundary,expanded_narrowest=args.expanded_narrowest)
