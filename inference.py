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

def parse_action(text):
    try:
        cleaned = text.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) >= 2:
            a = float(parts[0])
            b = float(parts[1])
            return np.array([max(-1.0, min(1.0, a)), max(-1.0, min(1.0, b))], dtype=np.float32)
    except Exception:
        pass
    return np.array([0.0, 0.0], dtype=np.float32)

def get_llm_action():
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Return exactly two comma-separated numbers in [-1, 1]."}
            ],
        )
        return parse_action(response.choices[0].message.content or "")
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return np.array([0.0, 0.0], dtype=np.float32)

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

        # Try LLM once, then reuse or fallback
        action = get_llm_action()

        for _ in range(20):
            try:
                obs, reward, done, truncated, info = env.step(action)
            except Exception as e:
                print(f"STEP error: {e}", file=sys.stderr)
                break

            steps += 1
            rewards.append(f"{reward:.2f}")
            is_done = done or truncated

            print(
                f"[STEP] step={steps} action={action.tolist()} reward={reward:.2f} "
                f"done={'true' if is_done else 'false'} error=null"
            )

            if isinstance(info, dict):
                success = bool(info.get("is_success", False) or info.get("goal_reached", False))

            if is_done:
                break

            # keep using same action or fallback
            action = action if action is not None else np.array([0.0, 0.0], dtype=np.float32)

    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
    finally:
        if env is not None:
            env.close()
        print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={','.join(rewards)}")

if __name__ == "__main__":
    run_inference()
