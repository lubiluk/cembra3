# Copied from https://github.com/araffin/rl-baselines-zoo
import gym
import numpy as np
import torch
import torchvision.transforms as transforms


class PreprocessingWrapper(gym.ObservationWrapper):
    """
    A wrapper that normalizes camera observations
    """

    def __init__(self, env):
        super(PreprocessingWrapper, self).__init__(env)

        self.img_size = (1, 200, 200)

        obs_spaces = dict(
            camera=gym.spaces.Box(
                -1.0,
                1.0,
                shape=self.img_size,
                dtype=np.float32,
            ),
            robot_state=env.observation_space.spaces["observation"]["robot_state"],
        )

        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=env.observation_space["desired_goal"],
                achieved_goal=env.observation_space["achieved_goal"],
                observation=gym.spaces.Dict(obs_spaces),
            )
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def observation(self, obs):
        """what happens to each observation"""

        # Convert image to grayscale
        img = obs["observation"]["camera"]

        obs["observation"]["camera"] = self.transform(img)

        return obs
