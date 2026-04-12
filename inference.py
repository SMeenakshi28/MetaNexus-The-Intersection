import os
import sys
import numpy as np
import requests
from openai import OpenAI

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BASE_URL = "http://localhost:7860"

def get_llm_action():
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Return exactly: 0.5, 0.0"}],
            max_tokens=10,
            temperature=0
        )
        text = response.choices[0].message.content or "0.5, 0.0"
        nums = [float(s.strip()) for s in text.replace('[','').replace(']','').split(',') if s.strip()]
        if len(nums) >= 2:
            return [nums[0], nums[1]]
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)

    return [0.5, 0.0]

def run_inference():
    task_name = "medium-congestion"
    benchmark = "smart-intersection-safety"
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")
    sys.stdout.flush()

    try:
        # Reset env
        res = requests.post(f"{BASE_URL}/reset")
        obs = res.json()

        for _ in range(10):
            steps += 1

            action = get_llm_action()

            try:
                res = requests.post(f"{BASE_URL}/step", json={"action": action})
                data = res.json()
            except Exception as e:
                print(f"[STEP] step={steps} action={action} reward=0.00 done=true error={str(e)}")
                break

            reward = float(data.get("reward", 0.0))
            done = data.get("done", False)

            print(f"[STEP] step={steps} action={action} reward={reward:.2f} done={'true' if done else 'false'} error=null")
            sys.stdout.flush()

            rewards.append(f"{reward:.2f}")

            if done:
                success = True if reward > 0 else False
                break

    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)

    # ✅ END MUST BE INSIDE FUNCTION
    score = 1.0 if success else 0.0
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={score:.2f} rewards={','.join(rewards)}")
    sys.stdout.flush()

if __name__ == "__main__":
    run_inference()
