from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from game_env import BekhaziGameEnv


def train_agent(timesteps: int = 10000):
    """
    Train a DQN agent to play the game on bekhazi.ca.
    The trained model will be saved to the current directory.
    """
    # Create environment. Use headless=False to see the browser while training.
    env = BekhaziGameEnv(headless=False)
    # Optionally check that the environment follows the Gym API
    # check_env(env)
    model = DQN('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save('bekhazi_agent')
    env.close()


if __name__ == '__main__':
    # Adjust number of timesteps as needed for training
    train_agent(1000)
