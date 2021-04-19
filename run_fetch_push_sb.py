import gym
import gym_fetch

from stable_baselines3 import HER
from stable_baselines3.sac import MlpPolicy
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper

env = gym.make("FetchPush-v2")

env = TimeFeatureWrapper(gym.make('FetchPush-v2'))

model = HER.load("data/fetch_push_sb", env=env)

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        obs = env.reset()