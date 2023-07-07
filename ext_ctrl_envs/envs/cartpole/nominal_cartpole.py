import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.optimize import minimize



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
    
    '''
    Helper Functions for calculating the reward
    '''
    def squared_exponential(self, x, ob):
        return -np.exp(-0.5 * (x - ob)**2)

    # Define the objective function to maximize
    def objective_function(self, x, ob):
        return self.squared_exponential(x, ob)


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
    

    # Step forward in simulation, given an action and trajectory
    def step_traj_track(self, a, x, x_dot, theta, theta_dot):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        '''
        Calculate reward based on max(squared exponential of tracking error)
        on trajectory state tracking
        '''
        result = minimize(self.objective_function, x, ob[0], method='BFGS')
        max_val_sq_exp_x = -result.fun

        result = minimize(self.objective_function, x_dot, ob[1], method='BFGS')
        max_val_sq_exp_x_dot = -result.fun

        result = minimize(self.objective_function, theta, ob[2], method='BFGS')
        max_val_sq_exp_theta = -result.fun

        result = minimize(self.objective_function, theta_dot, ob[3], method='BFGS')
        max_val_sq_exp_theta_dot = -result.fun

        
        reward = (max_val_sq_exp_x +
                  max_val_sq_exp_x_dot +
                  max_val_sq_exp_theta +
                  max_val_sq_exp_theta_dot)

        # termination state conditions
        # NOTE: the pendulum angle cutoff range is not considered
        terminated = bool(not np.isfinite(ob).all()) #or (np.abs(ob[1]) > 1.2)) 
        
        # to render or not to render
        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, False


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