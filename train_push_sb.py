import gym
import gym_fetch
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import HER
from sb3_contrib.common.wrappers.time_feature import TimeFeatureWrapper
from sb3_contrib.tqc import TQC
from sb3_contrib.tqc import MlpPolicy

env = TimeFeatureWrapper(gym.make('FetchPush-v2'))

policy_kwargs = dict(
    net_arch=[512, 512, 512],
    n_critics=2
)

model = HER(
    MlpPolicy,
    env,
    TQC,
    verbose=1,
    online_sampling=True,
    buffer_size=1_000_000,
    batch_size=2048,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    tau=0.05,
    goal_selection_strategy='future',
    n_sampled_goal=4,
    policy_kwargs=policy_kwargs,
)

model.learn(total_timesteps=1_000_000)
model.save("data/fetch_push_sb")

obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
