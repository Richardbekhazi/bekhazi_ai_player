from pathlib import Path
import time
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

from game_env import BekhaziGameEnv


def make_env(headless: bool):
    def _thunk():
        env = BekhaziGameEnv(headless=headless)
        return Monitor(env)
    return _thunk


def pick_newest_model_path() -> str:
    root = Path(".")
    candidates = []
    for name in ["bekhazi_agent.zip", "bekhazi_agent_interrupt.zip"]:
        p = root / name
        if p.exists():
            candidates.append(p)
    ckpt_dir = root / "checkpoints"
    if ckpt_dir.exists():
        candidates.extend(ckpt_dir.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError("No model found. Train once before running evaluate.")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading model file: {newest.name}. Modified at {time.ctime(newest.stat().st_mtime)}")
    return str(newest.with_suffix(""))

def main():
    model_path = pick_newest_model_path()

    base_env = DummyVecEnv([make_env(headless=False)])
    env = VecFrameStack(base_env, n_stack=4, channels_order="first")

    model = DQN.load(model_path)

    obs = env.reset()
    epsilon = 0.00

    while True:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        action = np.array([int(action)])  # vec env wants an array
        obs, reward, done, info = env.step(action)

        if done[0]:
            env.reset()

if __name__ == "__main__":
    main()
