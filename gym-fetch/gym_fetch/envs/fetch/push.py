import os
from gym import utils
from .. import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push.xml")
OBJECT_COUNT = 3
OBJECT_POS = [
    [1.25, 0.53, 0.425, 1.0, 0.0, 0.0, 0.0],
    [1.25, 0.53, 0.41, 1.0, 0.0, 0.0, 0.0],
    [1.25, 0.53, 0.41, 1.0, 0.0, 0.0, 0.0],
]
OBJECT_BOX = [(0.05, 0.05, 0.05), (0.2, 0.02, 0.02), (0.1, 0.1, 0.02)]


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            obj_count=OBJECT_COUNT,
            initial_obj_pos=OBJECT_POS,
            obj_bbox=OBJECT_BOX,
        )
        utils.EzPickle.__init__(self)
