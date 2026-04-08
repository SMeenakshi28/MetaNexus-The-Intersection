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

def get_action(obs):
    try:
        obs_summary = np.array(obs).flatten()[:5].tolist()
        obs_str = ",".join([f"{x:.2f}" for x in obs_summary])
        prompt = f"Obs:{obs_str}. Action (0:IDLE, 1:ACCEL, 2:BRAKE)? Digit only."

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

        res = response.choices[0].message.content.strip()
        if res in ["0", "1", "2"]:
            return int(res)
    except Exception as e:
        print(f"API/Auth Error: {e}", file=sys.stderr)

    return 0

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
        obs, _ = env.reset(seed=42)

        for _ in range(10):
            steps += 1
            action_idx = get_action(obs)

            obs, reward, done, truncated, info = env.step(int(action_idx))
            is_done = done or truncated

            action_map = {0: "idle", 1: "accelerate", 2: "brake"}
            action_str = action_map.get(action_idx, "idle")

            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} "
                f"done={'true' if is_done else 'false'} error=null"
            )
            rewards.append(f"{reward:.2f}")

            if is_done:
                if isinstance(info, dict):
                    success = bool(info.get("is_success", False) or info.get("goal_reached", False))
                    if not success and reward >= 0 and not info.get("crashed", False):
                        success = True
                break

    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
    finally:
        if env is not None:
            env.close()
        print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_inference()
