import numpy as np
from stable_baselines3 import PPO
from src.environment import CarRacingEnv

def main():
    env = CarRacingEnv()
    model_path = "models/ppo_car_agent.zip"
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    episodes = 20
    rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")

if __name__ == "__main__":
    main()
