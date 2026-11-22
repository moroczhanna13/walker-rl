import gymnasium as gym

import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# Létrehozzuk a környezetet: a szimulált világot a robottal
env = gym.make("Walker2d-v5")

# Milyen alakú adat megy ki és be?
print("Megfigyelési tér:", env.observation_space)
print("Akciótér:", env.action_space)

# reset(): új epizód indul, a robot álló helyzetbe kerül
obs, info = env.reset(seed=42)
print("Első megfigyelés:", obs)

# Egy véletlen akció: 6 szám -1 és 1 között
env.action_space.seed(42)
action = env.action_space.sample()
print("Véletlen akció:", action)

# step(): a szimulátor lép egyet az akcióval
#obs, reward, terminated, truncated, info = env.step(action)
#print("Jutalom:", reward)
#print("terminated:", terminated, "| truncated:", truncated)
#print("info:", info)

for i in range(50):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Lépés {i+1}: {obs[0]}")

env.close()