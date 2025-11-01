import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import time
import torch

# GPU Check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Model Setup
train_env = make_vec_env("CarRacing-v3", n_envs=16)


policy_kwargs = dict(
    features_extractor_kwargs=dict(features_dim=1024),  # larger feature space
    net_arch=[dict(pi=[512, 256], vf=[512, 256])]       # bigger actor/critic networks
)

model = PPO(
    policy="CnnPolicy",         
    env=train_env,
    verbose=1,
    device=device,              
    n_steps=2048,               
    batch_size=256,             
    learning_rate=3e-4,
    policy_kwargs=policy_kwargs,
)

# Model Training
model.learn(total_timesteps=10_000_000)
model.save("ppo_carracing2_gpu")
train_env.close()

# Model Evaluation and Rendering
render_env = gym.make("CarRacing-v3", render_mode="human")
model = PPO.load("ppo_carracing_gpu", device=device)


obs, info = render_env.reset()
done = False
total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = render_env.step(action)
    total_reward += reward
    render_env.render()

    if terminated or truncated:
        print(f"Episode finished — Total Reward: {total_reward:.2f}")
        time.sleep(1)
        obs, info = render_env.reset()
        total_reward = 0
