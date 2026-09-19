import threading

from fastapi import FastAPI, Request

from models.avoidance.wall_clear_policy import WallClearOrchardPolicy

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")
lock = threading.Lock()
last_time = -1.0
policy = None

TUNED = {
    "breed_reserve": 200.2397, "cap_hard_min": 1, "cap_max": 31,
    "cap_min": 2, "cap_mult": 0.5211, "cap_tree_slack": -1,
    "cluster_radius": 26.6189, "dist_pen": 0.166,
    "dump_after_t": 1869.3037, "dump_mult": 2.9442,
    "explore_energy": 252.9455, "extra_old": False, "feed_mode": "hungry",
    "fit_energy": 0.6062, "fit_hear": 0.571, "fit_speed": 0.2059,
    "fit_vision": 0.9802, "fruit_min_wait": 2.1084,
    "fruit_reach": 267.142, "heir_age": 55.5102, "heir_at_food": False,
    "heir_needs_site": True, "heir_reserve": 291.4568,
    "heir_select": False, "hungry_margin": 4.2384,
    "low_pop_reserve": 211.7621, "old_eat_last": False,
    "old_reach": 69.1117, "ripen_wait": 18.782, "rot_margin": 46.0607,
    "select_min_young": 2, "spread_weight": 1.0464,
    "sweep_rate": 0.0255, "tree_reach": 405.6673, "tree_slots": 1,
    "watch_patience": 17.669, "watch_reach": 376.7653,
}


def _new_policy():
    return WallClearOrchardPolicy(seed=0, **TUNED)

@app.post("/predict")
async def predict(request: Request):
    global policy, last_time
    body = await request.json()
    states = body.get("agent_status") or []
    for state in states:
        for obs in state.get("observations") or []:
            obs["type"] = str(obs.get("type", "")).capitalize()
    sim_time = body.get("sim_time")
    with lock:
        active = _new_policy() if sim_time is None else policy
        if sim_time is not None and (active is None or sim_time < last_time):
            active = policy = _new_policy()
        actions = active(states, float(sim_time or 0.0))
        if sim_time is not None:
            last_time = float(sim_time)
    return {"actions": [action.model_dump() if hasattr(action, "model_dump") else action.dict()
                        for _, action in actions]}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
