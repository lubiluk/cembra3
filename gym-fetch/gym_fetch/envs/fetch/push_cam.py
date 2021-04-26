import numpy as np

from .push import FetchPushEnv
from .. import rotations, robot_env, utils


class FetchPushCamEnv(FetchPushEnv):
    def __init__(self, reward_type="sparse"):
        super().__init__(reward_type=reward_type)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_name = 'object{}'.format(self._obj_idx)
            object_pos = self.sim.data.get_site_xpos(object_name)
        else:
            object_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        camera = self.sim.render(width=200, height=200, camera_name='external_camera_0')

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = {
            'robot_state': np.concatenate([
                    grip_pos, gripper_state, grip_velp, gripper_vel
                ]).copy(),
            'camera': camera.copy() 
        }

        return {
            'observation': obs,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
