from fastapi import FastAPI, Body
from threading import Lock
from src.utils.DTOs import StepResponse
from src.utils.controllers.expert_policy import ExpertPolicy

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")
policy = ExpertPolicy()
policy_lock = Lock()

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    with policy_lock:
        actions = [action.model_dump() for action in policy.actions_for_step(
            [agent.model_dump() for agent in step.agent_status],
            sim_time=step.sim_time, game_status=step.game_status,
        )]
    
    # Must return {"actions": [...]} format
    return {"actions": actions}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
