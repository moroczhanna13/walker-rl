import os

import gymnasium as gym
import imageio

# render_mode="rgb_array": a környezet képkockákat ad vissza, nem ablakot nyit
env = gym.make("Walker2d-v5", render_mode="human")
frames = []

for episode in range(5):
    obs, info = env.reset(seed=episode)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()      # véletlen akció
        #action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # álló helyzetben maradunk
        obs, reward, terminated, truncated, info = env.step(action)
        #frames.append(env.render())             # az aktuális képkocka eltárolása
        total_reward += reward
        steps += 1
        done = terminated or truncated          # elesett VAGY lejárt az idő

    print(f"{episode + 1}. epizód: {steps} lépés, összjutalom: {total_reward:.1f}")

env.close()

#os.makedirs("videos", exist_ok=True)
#imageio.mimsave("videos/before.gif", frames, fps=30, loop=0)
#print(f"Kész: {len(frames)} képkocka -> videos/before.gif")