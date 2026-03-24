import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from src.environment import CarRacingEnv
import os
import shutil

def main():
    model_path = "models/ppo_car_agent.zip"
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tmp_dir = "results/video_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    env = CarRacingEnv(render_mode="rgb_array")
    # RecordVideo limits to specific episodes, here 0 since we run 1 episode.
    env = RecordVideo(env, video_folder=tmp_dir, episode_trigger=lambda x: True, name_prefix="demo")
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    env.close()

    if not os.path.exists("results"):
        os.makedirs("results")

    for file in os.listdir(tmp_dir):
        if file.endswith(".mp4"):
            shutil.move(os.path.join(tmp_dir, file), "results/agent_demonstration.mp4")
            break
            
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass

if __name__ == "__main__":
    main()
