from fastapi import FastAPI, Body
from threading import Lock
from src.utils.DTOs import StepResponse
from models.optimization_policy import OptimizationPolicy

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")
policy = OptimizationPolicy()
last_time = None
policy_lock = Lock()

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    global policy, last_time
    with policy_lock:
        if last_time is not None and step.sim_time < last_time:
            policy = OptimizationPolicy()
        states = [agent.model_dump() for agent in step.agent_status]
        for state in states:
            for observation in state["observations"]:
                if isinstance(observation.get("type"), str):
                    observation["type"] = observation["type"].capitalize()
        actions = [action.model_dump() for _, action in policy(states, step.sim_time)] if states else []
        last_time = step.sim_time
        if step.game_status == "game_over":
            policy, last_time = OptimizationPolicy(), None

    # Must return {"actions": [...]} format
    return {"actions": actions}

@app.get("/")
def index():
    return {"message": "Non-trapping agent endpoint running", "policy": "OptimizationPolicy"}

if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Serve the non-trapping shared-map policy over HTTP")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
