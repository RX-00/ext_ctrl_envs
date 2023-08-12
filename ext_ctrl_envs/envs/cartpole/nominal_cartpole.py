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
                          (np.abs(ob[1]) > 0.01) or  # keep pend upright
                          (np.abs(ob[0]) > 0.01))    # keep cart at near origin
        
        if self.render_mode == "human":
            self.render()

        if not terminated:
            reward = 1.0
        
        return ob, reward, False, False, {}


    # Calculate reward
    def calc_reward(self, ref_val, obs_val, weight_h, alpha):
        tracking_reward = (weight_h * np.exp(-alpha * np.abs(ref_val - obs_val)**2))
        if np.abs(tracking_reward) > weight_h:
            print("ERR: tracking_reward exceeded weight_h, this shouldn't be possible")
        return np.abs(tracking_reward)
    

    # Step forward in simulation, given an action and trajectory
    def step_traj_track(self, a, x, x_dot, theta, theta_dot, u, weight_h, interm_weights):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = 0
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

        for indx, interm_weight in enumerate(interm_weights):
            if (indx == 0 or indx == 1):
                # punish for not being cart 0 and pend 0
                # punish 10 comes from 5 terms in state vector times 2
                #reward -= 10 * np.abs(obs[indx] - 0.0)**2
                punish = 4 # 10
                reward -= -punish * np.exp(-10*np.abs(obs[indx])**2) + punish
                # reward for being close to trajectory
                reward += self.calc_reward(ref_val=0.0,
                                           obs_val=obs[indx],
                                           weight_h=3,
                                           alpha=100) # increase to make it more precise to earn reward
                        
            if (indx < 4): # 4
                reward += self.calc_reward(ref_val=ref_obs[indx],
                                            obs_val=obs[indx], 
                                            weight_h=weight_h,
                                            alpha=interm_weight)

        #reward += self.calc_reward(ref_val=0.0,
        #                           obs_val=np.abs(obs[0])+np.abs(obs[1]),
        #                           weight_h=2,
        #                           alpha=1000) # increase to make it more precise to earn reward
        
        # termination conditions
        if (not np.isfinite(obs).all() or #r_trunc > epsilon):
            np.abs(ref_obs[0] - obs[0]) > 0.25 or # 0.25 is too tight
            np.abs(ref_obs[1] - obs[1]) > 0.25 or # 0.25 is too tight
            #np.abs(ref_obs[2] - obs[2]) > 0.25 or # 0.25 is too tight
            #np.abs(ref_obs[3] - obs[3]) > 0.25 or # 0.25 is too tight
            np.abs(obs[1]) > np.pi):
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