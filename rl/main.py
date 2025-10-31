import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import time
import torch

# ✅ Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1️⃣ Create vectorized training environments (parallel simulation for speed)
# CarRacing is heavy, so 4–8 parallel envs is ideal
train_env = make_vec_env("CarRacing-v3", n_envs=32)

# 2️⃣ Create PPO model with a larger CNN policy for better GPU utilization
policy_kwargs = dict(
    features_extractor_kwargs=dict(features_dim=1024),  # larger feature space
    net_arch=[dict(pi=[512, 256], vf=[512, 256])]       # bigger actor/critic networks
)

model = PPO(
    policy="CnnPolicy",         # use CNN policy for image input
    env=train_env,
    verbose=1,
    device=device,              # <--- GPU support here
    n_steps=2048,               # longer rollout buffer for stable training
    batch_size=256,             # larger batch to utilize GPU better
    learning_rate=3e-4,
    policy_kwargs=policy_kwargs,
)

# 3️⃣ Train the model
# You can increase to 1_000_000+ timesteps later
model.learn(total_timesteps=100_000_000)
model.save("ppo_carracing2_gpu")
train_env.close()

# 4️⃣ Create render environment
render_env = gym.make("CarRacing-v3", render_mode="human")

# 5️⃣ Load trained model onto GPU
model = PPO.load("ppo_carracing_gpu", device=device)

# 6️⃣ Render the trained agent
obs, info = render_env.reset()
done = False
total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = render_env.step(action)
    total_reward += reward
    render_env.render()

    if terminated or truncated:
        print(f"🏁 Episode finished — Total Reward: {total_reward:.2f}")
        time.sleep(1)
        obs, info = render_env.reset()
        total_reward = 0
