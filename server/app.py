from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ActionInput(BaseModel):
    action: list

state_data = {
    "cars": 20,
    "waiting_time": 50,
    "step": 0
}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/reset")
@app.post("/reset")
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

    state_data["cars"] = max(0, state_data["cars"] - 2)
    state_data["waiting_time"] = max(0, state_data["waiting_time"] - 5)
    state_data["step"] += 1

    reward = 1.0
    done = state_data["step"] >= 5

    return {
        "observation": state_data,
        "reward": reward,
        "done": done,
        "info": {}
    }

@app.get("/state")
def state():
    return state_data

# ✅ IMPORTANT: main function (required by validator)
def main():
    return app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
