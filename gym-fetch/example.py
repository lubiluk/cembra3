import gym
import gym_fetch
import numpy as np

env = gym.make("FetchPush-v2")
env.reset()
env.render()

for _ in range(1000):
    action = np.random.sample(4) * 2 - 1
    o, r, d, i = env.step(action)
    env.render()

    if d:
        env.reset()

env.close()
