import os

import gym
import gym_fetch
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos import SAC
from algos.common import replay_buffer_her_cam
from algos.sac import core_her_cam
from utils.wrappers import PreprocessingWrapper

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

save_path = "./data/push_cam_su"
exp_name = "push_cam_su"

os.makedirs(save_path, exist_ok=True)

env = PreprocessingWrapper(gym.make("FetchPushCam-v2"))


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        obs_space = env.observation_space.spaces["observation"]["camera"]

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 8, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            obs = torch.as_tensor(obs_space.sample()[None]).float()
            n_flatten = self.cnn(obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())

    def forward(self, x):
        x = self.linear(self.cnn(x))
        return x


ac_kwargs = dict(
    hidden_sizes=[256, 256], activation=nn.ReLU, extractor_module=Extractor
)
rb_kwargs = dict(size=10_000,
                 n_sampled_goal=4,
                 goal_selection_strategy='future')

logger_kwargs = dict(output_dir=save_path, exp_name=exp_name)

model = SAC(
    env=env,
    actor_critic=core_her_cam.MLPActorCritic,
    ac_kwargs=ac_kwargs,
    replay_buffer=replay_buffer_her_cam.ReplayBuffer,
    rb_kwargs=rb_kwargs,
    max_ep_len=100,
    batch_size=256,
    gamma=0.95,
    lr=0.0003,
    update_after=512,
    update_every=512,
    logger_kwargs=logger_kwargs,
    use_gpu_buffer=True,
)

model.train(steps_per_epoch=1024, epochs=5000)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env(save_path)

run_policy(env, get_action)
