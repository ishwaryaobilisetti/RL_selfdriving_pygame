# RL Car Racing - 2D Sensor-Based End-To-End Pipeline

A complete Reinforcement Learning ecosystem built to train an autonomous car in a bespoke Pygame-based 2D racing environment.

## Overview
Unlike heavy RL tasks operating on pixel frames, this environment prioritizes efficiency and lightweight compute explicitly. It operates using ray-cast "LIDAR-style" distance sensors to observe the track walls. The Gym environment shapes the rewards to drive stable convergence using Proximal Policy Optimization (PPO) via `stable-baselines3`. 

## Features
- **Custom Tracking Engine:** A self-contained Pygame kinematics solver with collision detection.
- **Gymnasium API Wrapper:** Standards-compliant environment with `spaces.Box` observation spaces and `spaces.Discrete` actions.
- **Training Logging and Metrics:** Extracts model metrics dynamically, natively producing `training_log.json` and a visualization artifact out of the box.

## How to Run Locally

1. Install Dependencies:
```bash
pip install -r requirements.txt
```

2. Training:
```bash
python train.py
```
This process generates `models/ppo_car_agent.zip` alongside logging graphs.

3. Evaluation:
```bash
python evaluate.py
```
This loads your model and provides `Mean Reward` mapping via `STDOUT`.

4. Visual Demonstration (MP4):
```bash
python record_video.py
```

## Docker Containerization
You can spin this process up inside a container natively avoiding environment complexity, using the provided `Dockerfile`.
```bash
docker build -t car-agent .
```bash
docker run car-agent python evaluate.py
```

### Running with Docker Compose

You can easily run different stages of the RL pipeline using Docker Compose:

- **Evaluate the agent:**
  ```bash
  docker compose run evaluate
  ```
- **Train the agent (Long process):**
  ```bash
  docker compose run train
  ```
- **Record a new demonstration video:**
  ```bash
  docker compose run record
  ```

## Setup Instructions & Analysis
The agent rapidly identifies optimal paths through continuous, heavily penalized boundary collision incentives (`-10.0`), supported by high-frequency raycasting sensing across eight normalized vectors. It balances precision versus pure throughput via explicit time penalties (`-0.01`). Checkpoints provide structured intermediate rewards (`+1.0`) forcing adherence to track mapping. Testing via configuration overrides (`config.yaml`) enables iteration across steps and epochs.
