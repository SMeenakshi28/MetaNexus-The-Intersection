import os
import sys
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

def run_inference():
    task_name = "medium-congestion"
    benchmark = "smart-intersection-safety"
    rewards = []
    steps = 0
    success = False
    env = None

    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")

    try:
        env = gym.make("intersection-v1")

        # Debug info to stderr only
        print(f"action_space={env.action_space}", file=sys.stderr)
        print(f"observation_space={env.observation_space}", file=sys.stderr)

        obs, _ = env.reset(seed=42)
        print(f"obs_type={type(obs)}", file=sys.stderr)
        print(f"obs_shape={getattr(obs, 'shape', None)}", file=sys.stderr)

        # Correct action for Box(-1.0, 1.0, (2,), float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        print(f"trying action={action.tolist()}", file=sys.stderr)

        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        is_done = done or truncated

        print(
            f"[STEP] step={steps} action={action.tolist()} reward={reward:.2f} "
            f"done={'true' if is_done else 'false'} error=null"
        )
        rewards.append(f"{reward:.2f}")

        if isinstance(info, dict):
            success = bool(info.get("is_success", False) or info.get("goal_reached", False))

    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
    finally:
        if env is not None:
            env.close()
        print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_inference()
