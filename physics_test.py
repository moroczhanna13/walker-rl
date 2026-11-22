import argparse

import gymnasium as gym
import numpy as np

from common import load_policy
from wrappers import PhysicsWrapper

DEFAULT_VALUES = {
    "mass": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "friction": [0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0],
    "motor": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
}

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Walker2d-v5")
parser.add_argument("--model", required=True)
parser.add_argument("--param", required=True, choices=["mass", "friction", "motor"], help="melyik fizikai paramétert változtassa")
parser.add_argument("--values", type=float, nargs="+", default=None, help="szorzók; 1.0 = eredeti")
parser.add_argument("--episodes", type=int, default=20)
args = parser.parse_args()

values = args.values or DEFAULT_VALUES[args.param]
policy = load_policy(args.model, args.env)
print(f"Modell: {args.model} | változó: {args.param} | {args.episodes} epizód értékenként")
print(f"{'Szorzó':>8} {'Elesett':>10} {'Átl. jutalom':>13} {'Átl. hossz':>11} {'Sebesség':>10}")

for value in values:
    env = PhysicsWrapper(gym.make(args.env), **{args.param: value})
    dt = env.unwrapped.dt
    returns, lengths, speeds, falls = [], [], [], 0
    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)
        start_x = env.unwrapped.data.qpos[0]
        total, steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        speeds.append((env.unwrapped.data.qpos[0] - start_x) / (steps * dt))
        falls += int(terminated)
    env.close()
    print(f"{value:8.2f} {falls:>6}/{args.episodes:<3} {np.mean(returns):13.0f} {np.mean(lengths):11.0f} {np.mean(speeds):8.2f} m/s")