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
        reward = 0.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        
        # termination state conditions
        terminated = bool(not np.isfinite(ob).all() or 
                          (np.abs(ob[1]) > 0.1) or  # keep pend upright
                          (np.abs(ob[0]) > 0.1))    # keep cart at near origin
        
        if self.render_mode == "human":
            self.render()
        
        if not terminated:
            reward = 1.0

        return ob, reward, terminated, False, {}


    # Calculate reward
    def calc_reward(self, ref_val, obs_val, weight_h, alpha):
        tracking_reward = (weight_h * np.exp(-alpha * abs(ref_val - obs_val)**2))
        if tracking_reward > weight_h:
            print("ERR: tracking_reward exceeded weight_h, this shouldn't be possible")
        return tracking_reward
    

    # Step forward in simulation, given an action and trajectory
    def step_traj_track(self, a, x, x_dot, theta, theta_dot, u, weight_h, interm_weights):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = 0
        gamma = 0
        terminated = False
        '''
        Calculate reward based on max(squared exponential of tracking error)
        on trajectory state tracking

        # NOTE: ob is organized as follows,
                [x, theta, x_dot, theta_dot], we add action a to get
                obs
                [x, theta, x_dot, theta_dot, a]
                 0    1      2      3        4
        '''
        obs = np.append(ob, a)
        ref_obs = [x, theta, x_dot, theta_dot, u]
        indx = 0
        for interm_weight in interm_weights:
            reward += self.calc_reward(ref_val=ref_obs[indx],
                                       obs_val=obs[indx], 
                                       weight_h=weight_h,
                                       alpha=interm_weight)
            indx += 1

        # truncation criterion
        for i in range(obs.size):
            gamma += 1.0 / (3.0 * obs.size) * abs(ref_obs[i] - obs[i])
        epsilon = 0.5
        r_trunc = 1 - gamma / epsilon
        
        # termination conditions
        if (not np.isfinite(obs).all() or
            np.abs(obs[1]) > 1.2 or
            r_trunc > epsilon):
            terminated = True

        if terminated:
            reward = 0
        # to render or not to render
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
    
    # Return state observation
    def get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()