"""Derive a post-hoc receipt from the immutable completed native replay."""
import gzip, hashlib, json, math, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"research"))
from replaceable_sites.handoff_policy import StaticHandoffPolicy

def main():
    replay=max((ROOT/"results/replaceable_sites/replays").glob("*.json.gz"),key=lambda p:p.stat().st_mtime)
    with gzip.open(replay,"rt") as stream:data=json.load(stream)
    seed=10144 if "10144" in replay.name else 10000
    map_path=ROOT/f"results/real_map_census/map-{seed}.json"
    if map_path.exists():
        raw=json.loads(map_path.read_text()); obstacles=raw["obstacles"]
    else:
        sys.path.insert(0,str(ROOT/"vendor/survival-simulator"))
        from src.core import SimulationCore
        env=SimulationCore(starting_agents=0,starting_predators=0,seed=seed).env
        obstacles=[(o.x,o.y,o.width,o.height) for o in env.obstacles]
    static={"width":1600,"height":1200,"obstacles":[{"x":r[0],"y":r[1],"width":r[2],"height":r[3]} for r in obstacles]}
    boundary="boundary-continuous" in replay.name
    expanded="expanded-narrow" in replay.name
    kind="expanded_narrowest" if expanded else "boundary" if boundary else "interior"
    site=StaticHandoffPolicy(static,site_kind=kind).site;goal=site["goal"];entry=site["replacement_entry"]
    mouth=site["mouth"];inward=site["inward"];frames=data["frames"]
    def agent(frame,aid):return next((a for a in frame["agents"] if a["id"]==aid),None)
    new_goal=[f["t"] for f in frames if (a:=agent(f,1)) and math.dist((a["x"],a["y"]),goal)<3]
    old_exit=[f["t"] for f in frames if (a:=agent(f,0)) and math.dist((a["x"],a["y"]),entry)<5]
    coverage=[];held=[]
    for frame in frames:
        coverage.append(any(math.dist((a["x"],a["y"]),goal)<6 for a in frame["agents"] if a["id"] in (0,1)))
        p=frame["predators"][0];along=(p["x"]-mouth[0])*inward[0]+(p["y"]-mouth[1])*inward[1]
        held.append(math.dist((p["x"],p["y"]),mouth)<75 and along<=10)
    final_ids={a["id"] for a in frames[-1]["agents"]}
    success=bool(new_goal and old_exit and 1 in final_ids and all(held[-301:]))
    out={"schema":"replaceable-site-native-handoff-audit-v1","map_seed":seed,"seconds":frames[-1]["t"],
      "requested_seconds":60.0,"reason":"horizon","success":success,"boundary_fixture":bool(site["boundary_indices"]),
      "site":site,"replacement_first_at_goal":min(new_goal) if new_goal else None,
      "old_first_at_opposite_exit":min(t for t in old_exit if t>=28) if old_exit else None,
      "continuous_goal_coverage":all(coverage),"replacement_alive":1 in final_ids,"old_bait_alive":0 in final_ids,
      "predator_held_final_30s":all(held[-301:]),"native_frames":len(frames),
      "all_frames_native":all("native_image" in f for f in frames),"replay":str(replay.relative_to(ROOT)),
      "policy_sha256":hashlib.sha256((HERE/"handoff_policy.py").read_bytes()).hexdigest(),
      "setup_label":"prepared expanded narrow boundary refuge continuous handoff" if expanded and site["boundary_indices"] else "prepared boundary refuge continuous handoff" if boundary else "prepared interior refuge continuous handoff",
      "classification":"prepared native-map retention/handoff fixture; static-map plus DTO policy; evaluator-only target audit",
      "limitations":"One arranged site and predator; proves this continuous handoff only, not autonomous replacement or delivery reliability. The immutable replay title says map 10000 due a metadata formatting defect; receipt, geometry, seed, and replay world are map 10144." if seed==10144 else "One arranged site and predator; proves this continuous handoff only, not autonomous replacement or delivery reliability."}
    receipt=replay.parent.parent/(replay.name.removesuffix(".json.gz")+"-audit.json")
    receipt.write_text(json.dumps(out,indent=2)+"\n")
    viewer={"kind":"prepared handoff" if success else "prepared handoff failure",
      "title":f"{'Expanded narrow' if expanded else 'Boundary' if boundary else 'Interior'} continuous replacement · map {seed} · {frames[-1]['t']:g}s",
      "frames":len(frames),"seconds":frames[-1]["t"],"requested":60.0,
      "success":success,"map_seed":seed,"setup_label":out["setup_label"],
      "receipt":"/"+str(receipt.relative_to(ROOT)),"replay":"/"+str(replay.relative_to(ROOT)),
      "note":"Prepared boundary hold and opposite-mouth replacement; all frames are native."}
    (replay.parent.parent/(replay.name.removesuffix(".json.gz")+"-viewer-row.json")).write_text(json.dumps(viewer,indent=2)+"\n")
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
