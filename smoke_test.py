from game_env import BekhaziGameEnv
import numpy as np
import time

env = BekhaziGameEnv(headless=False)
obs, info = env.reset()
print("obs shape:", obs.shape)

for t in range(200):
    action = np.random.randint(0, 3)
    obs, reward, terminated, truncated, info = env.step(action)
    if t % 20 == 0:
        print(f"t={t} score={info.get('score')} reward={reward:.3f}")
    if terminated or truncated:
        print("episode ended. restarting")
        obs, info = env.reset()
        time.sleep(0.3)

env.close()
