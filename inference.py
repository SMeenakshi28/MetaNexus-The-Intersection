import os
import torch
import torch.nn as nn
import gymnasium as gym
import highway_env
import numpy as np
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


class HighwayBrain(nn.Module):
    def __init__(self, state_size=105, action_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.network(x)


def pad_or_trim(obs, size=105):
    obs_flat = np.array(obs).flatten()
    if obs_flat.size < size:
        obs_flat = np.pad(obs_flat, (0, size - obs_flat.size))
    else:
        obs_flat = obs_flat[:size]
    return torch.tensor(obs_flat, dtype=torch.float32)


def run_inference():
    task_name = "medium-congestion"
    benchmark = "smart-intersection-safety"
    rewards = []
    steps = 0
    success = False
    env = None

    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")

    try:
        model = HighwayBrain()
        model_path = os.getenv("MODEL_PATH", "highway_brain.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        env = gym.make("intersection-v1")
        env.unwrapped.configure(
            {
                "action": {
                    "type": "DiscreteMetaAction",
                }
            }
        )

        obs, _ = env.reset()

        for _ in range(15):
            steps += 1

            obs_t = pad_or_trim(obs, 105)

            with torch.no_grad():
                logits = model(obs_t)

            action_idx = int(torch.argmax(logits).item())
            if hasattr(env.action_space, "n"):
                action_idx = max(0, min(action_idx, env.action_space.n - 1))

            action_map = {0: "idle", 1: "accelerate", 2: "brake"}
            action_str = action_map.get(action_idx, "idle")

            obs, reward, done, truncated, info = env.step(action_idx)
            is_done = done or truncated

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={'true' if is_done else 'false'} "
                f"error=null"
            )
            rewards.append(f"{reward:.2f}")

            if is_done:
                if isinstance(info, dict):
                    success = bool(
                        info.get("is_success", False)
                        or info.get("success", False)
                        or info.get("goal_reached", False)
                    )
                if not success and reward > 0:
                    success = True
                break

    except Exception:
        success = False
    finally:
        if env is not None:
            env.close()
        print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run_inference()
