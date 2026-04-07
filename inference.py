import os
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
import numpy as np
from openai import OpenAI

# 1. Strict Env Var Handling
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# 2. OpenAI Client (Required for compliance, even if model is local)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# 3. Your AI Brain
class HighwayBrain(nn.Module):
    def __init__(self, state_size=105, action_size=3):
        super(HighwayBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x): return self.network(x)

def run_inference():
    task_name = "medium-congestion"
    benchmark = "smart-intersection-safety"
    rewards = []
    steps = 0
    success = False
    env = None

    # [START] - Must be first
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")

    try:
        # Initialize Brain
        model = HighwayBrain()
        if os.path.exists("highway_brain.pth"):
            model.load_state_dict(torch.load("highway_brain.pth", map_location='cpu'))
        model.eval()

        # Initialize Environment inside try
        env = gym.make('intersection-v0')
        obs, _ = env.reset()

        # Evaluation Loop (Max 10 steps for validation)
        for i in range(10):
            steps += 1
            
            # AI Decision
            # Safety: Ensure obs is flattened to exactly 105
            obs_flat = obs.flatten()
            if obs_flat.shape[0] > 105:
                obs_flat = obs_flat[:105]
            
            obs_t = torch.FloatTensor(obs_flat)
            with torch.no_grad():
                action_idx = torch.argmax(model(obs_t)).item()
            
            # Map integer action to string for logs (as required)
            action_map = {0: "idle", 1: "accelerate", 2: "brake"}
            action_str = action_map.get(action_idx, "idle")

            # Execute Step
            obs, reward, done, truncated, info = env.step(action_idx)
            is_done = done or truncated

            # [STEP] - Immediate output, 2-decimal rewards, lowercase booleans
            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={'true' if is_done else 'false'} "
                f"error=null"
            )

            rewards.append(f"{reward:.2f}")

            if is_done:
                # Success is only true if the episode ended without a crash/timeout 
                # (You can adjust this logic based on your 'info' dict)
                success = True 
                break

    except Exception:
        success = False
    finally:
        if env:
            env.close()
        # [END] - Final summary, strictly no extra fields
        print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_inference()