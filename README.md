# Smart-Intersection-Safety (Meta OpenEnv)

An Autonomous Traffic Management environment designed to minimize wait times and prevent collisions at uncontrolled 4-way intersections using Reinforcement Learning.

## 🚀 Environment Description
This environment simulates real-world vehicle-to-vehicle (V2V) interactions. The agent must control the ego-vehicle to navigate through cross-traffic safely and efficiently.

## 📊 Specification
- **Observation Space**: The environment returns a JSON-based state:
json

{
  "cars": int,
  "waiting_time": int,
  "step": int
}
- **Action Space**: Continuous (list of 2 floats)  
  Example: [0.5, 0.0]
- **Tasks**: 
  - `easy-flow`: Single vehicle interaction.
  - `medium-congestion`: Moderate urban traffic (5 vehicles).
  - `hard-peak-hour`: High-density complex interactions (12 vehicles).

## 🛠️ Setup & Reproducibility
1. **Local Run**: 
   - `pip install -r requirements.txt`
   - `streamlit run app.py`
2. **Docker**:
   - `docker build -t traffic-ai .`
   - `docker run -p 7860:7860 traffic-ai`

## 🎯 Reward Function
The agent receives a dense reward signal:
- **Efficiency**: +1.0 for maintaining target speed.
- **Safety**: -5.0 penalty for near-misses.
- **Collision**: -10.0 penalty (terminates episode).
- **Progress**: Normalized 0.0-1.0 score based on average step-wise efficiency.
