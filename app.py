from fastapi import FastAPI
import gymnasium as gym
import highway_env
import numpy as np

app = FastAPI()

env = gym.make("intersection-v1")
env.unwrapped.configure({
    "action": {"type": "ContinuousAction"}
})

obs, _ = env.reset(seed=42)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/reset")
def reset():
    global obs
    obs, _ = env.reset(seed=42)
    return {"observation": obs.tolist()}

@app.post("/step")
def step(action: dict):
    global obs
    act = np.array(action.get("action", [0.5, 0.0]), dtype=np.float32)

    obs, reward, done, truncated, info = env.step(act)

    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(done or truncated),
        "info": info
    }

@app.get("/state")
def state():
    return {"observation": obs.tolist()}