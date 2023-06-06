# testing script for nonnominal cartpole environment

# PyTorch
# NOTE: you must import torch before mujoco or else there's an invalid pointer error
#       - source: https://github.com/deepmind/mujoco/issues/644
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import mujoco
import mediapy as media
import matplotlib.pyplot as plt

# Mujoco and custom environments
import gymnasium as gym
import ext_ctrl_envs

# Create the env
env_id = "NonnonimalCartpole"
env = gym.make(env_id, render_mode="human")

# State of system and whether or not to render env
# reset() returns a tuple of the observation and nothing
state = env.reset()[0] # np.array([ x, x_dot, theta, theta_dot ])
env.render()

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# recording info of the system simulation
x_positions = [state[0]]
theta_positions = [state[2]]

done = False

'''
NOTE: step != timestep, please refer to the .xml file for the simulation timestep
      as that would effect the energy in the system.
'''
#while not done: # while loop for training
for i in range(500): # for testing, 200 steps

    # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
    # These represent the next observation, the reward from the step,
    # if the episode is terminated, if the episode is truncated and
    # additional info from the step
    state, reward, terminated, truncated, info = env.step(action=[0])

    # record data about system
    x_positions.append(state[0])
    theta_positions.append(state[2])

    # End the episode when either truncated or terminated is true
    #  - truncated: The episode duration reaches max number of timesteps
    #  - terminated: Any of the state space values is no longer finite.
    done = terminated or truncated

env.close()

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(x_positions)
axs[0].set_title('Cart position')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Position')

fig.suptitle('Cartpole x and theta pos', fontsize=16)

axs[1].plot(theta_positions)
axs[1].set_xlabel('Time step')
axs[1].set_title('Pendulum angular position')
axs[1].set_ylabel('Radians')

plt.show()