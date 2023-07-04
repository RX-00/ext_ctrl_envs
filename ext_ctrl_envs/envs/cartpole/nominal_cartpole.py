import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class NominalCartpoleEnv(MujocoEnv, utils.EzPickle):

    '''
        NOTE: from [https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/inverted_pendulum_v4.py]
    '''

    # mandatory info for rendering env
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "offscreen",
        ],
        "render_fps": 50, # NOTE: This depends on the .xml's timestep value
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        xml_path = os.path.join(os.path.dirname(__file__), "nominal_cartpole.xml")
        MujocoEnv.__init__(
            self,
            xml_path,
            2, # frame_skip
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    # Step forward in simulation, given an action
    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        
        # termination state conditions
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))
        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    # Initial state of the system
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    # Return state observation
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()