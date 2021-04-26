import gym
import gym_fetch
import numpy as np
import cv2

env = gym.make("FetchPushCam-v2")
env.reset()
env.render()

for _ in range(1000):
    action = np.random.sample(4) * 2 - 1
    o, r, d, i = env.step(action)
    env.render()

    cv2.imshow("camera", o["observation"]["camera"])
    cv2.waitKey(1)

    if d:
        env.reset()

env.close()
