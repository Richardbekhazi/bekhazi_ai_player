import os
import signal
import sys
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from game_env import BekhaziGameEnv

torch.set_num_threads(2)  # keep CPU cooler


def make_env(headless: bool):
    def _thunk():
        env = BekhaziGameEnv(headless=headless)
        return Monitor(env)
    return _thunk


def main():
    # small but meaningful run
    total_timesteps = 8000
    headless = True

    base_env = DummyVecEnv([make_env(headless)])
    # match evaluate.py
    env = VecFrameStack(base_env, n_stack=4, channels_order="first")

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=500,
        batch_size=32,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        verbose=1,
        tensorboard_log="./tb_logs",
        device="cpu",
    )

    ckpt = CheckpointCallback(save_freq=5_000, save_path="./checkpoints", name_prefix="bekhazi_ckpt")

    def cleanup_and_exit(signum=None, frame=None):
        try:
            model.save("bekhazi_agent_interrupt")
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        print("Stopped. Saved as bekhazi_agent_interrupt.zip")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)

    try:
        model.learn(total_timesteps=total_timesteps, callback=ckpt)
        model.save("bekhazi_agent")
        print("Training complete. Saved as bekhazi_agent.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
