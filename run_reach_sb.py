import gym
import gym_fetch

from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy

env = gym.make("FetchReach-v2")

model = HER.load("data/fetch_reach_sb", env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()