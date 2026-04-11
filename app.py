from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Input model
class ActionInput(BaseModel):
    action: list

# Simple deterministic state
state_data = {
    "cars": 20,
    "waiting_time": 50,
    "step": 0
}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/reset")
def reset():
    global state_data
    state_data = {
        "cars": 20,
        "waiting_time": 50,
        "step": 0
    }
    return state_data

@app.post("/step")
def step(action: ActionInput):
    global state_data

    try:
        # Update state (deterministic)
        state_data["cars"] = max(0, state_data["cars"] - 2)
        state_data["waiting_time"] = max(0, state_data["waiting_time"] - 5)
        state_data["step"] += 1

        # Reward logic
        reward = 1.0 if action.action == "east" else 0.5

        done = state_data["step"] >= 5

        return {
            "observation": state_data,
            "reward": reward,
            "done": done,
            "info": {}
        }

    except Exception as e:
        return {
            "observation": state_data,
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }

@app.get("/state")
def state():
    return state_data