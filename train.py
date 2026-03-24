import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from src.environment import CarRacingEnv

class LoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(LoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.timesteps = []
        self.mean_rewards = []
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0.0
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(float(mean_reward))
            
            with open(os.path.join(self.log_dir, "training_log.json"), "w") as f:
                json.dump({"timesteps": self.timesteps, "mean_rewards": self.mean_rewards}, f, indent=2)
        return True

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Verify environment
    env = CarRacingEnv()
    check_env(env)
    
    env = Monitor(env) # Adds episode info ('r', 'l') to info dict
    
    # Setup model
    ppo_params = config["ppo_params"]
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=ppo_params["learning_rate"],
        n_steps=ppo_params["n_steps"],
        batch_size=ppo_params["batch_size"],
        gamma=ppo_params["gamma"],
        verbose=1
    )
    
    # Train
    callback = LoggingCallback(check_freq=2000, log_dir="results")
    model.learn(total_timesteps=config["training"]["total_timesteps"], callback=callback)
    
    # Save model
    model.save("models/ppo_car_agent")
    
    # Plot results
    log_path = "results/training_log.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
        plt.figure()
        plt.plot(data["timesteps"], data["mean_rewards"])
        plt.title("Training Reward Curve")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.savefig("results/reward_curve.png")

if __name__ == "__main__":
    main()
