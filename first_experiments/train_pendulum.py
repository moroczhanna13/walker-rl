import time

import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "InvertedPendulum-v5"
TOTAL_TIMESTEPS = 50_000

env = gym.make(ENV_ID)

# MlpPolicy (neurális háló, 2 réteg, 64-64 neuron)
model = PPO("MlpPolicy", env, verbose=1, seed=0, tensorboard_log="logs")

start = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="pendulum_ppo")
print(f"Tanítás kész: {time.time() - start:.0f} másodperc")

model.save("models/pendulum_ppo")
print("Modell elmentve: models/pendulum_ppo.zip")

env.close()