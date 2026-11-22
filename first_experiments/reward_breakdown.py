import gymnasium as gym

env = gym.make("Walker2d-v5")
obs, info = env.reset(seed=0)

print("lépés | magasság | dőlés | túlélés | előre  | motor   | jutalom")
done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    done = terminated or truncated
    if step % 3 == 0 or done:      # minden 3. lépést és az utolsót kiírja
        print(f"{step:5d} | {obs[0]:8.2f} | {obs[1]:5.2f} | "
              f"{info['reward_survive']:7.1f} | {info['reward_forward']:6.2f} | "
              f"{info['reward_ctrl']:7.4f} | {reward:7.2f}")

print()
if terminated:
    print(f"Vége a(z) {step}. lépésben: a robot elesett.")
    print(f"  törzs magassága: {obs[0]:.2f} m (megengedett: 0,8 és 2,0 között)")
    print(f"  törzs dőlése:    {obs[1]:.2f} rad (megengedett: -1 és 1 között)")
else:
    print(f"Vége a(z) {step}. lépésben: lejárt az időkorlát.")

env.close()