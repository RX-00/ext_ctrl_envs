# testing script for nominal cartpole environment

import numpy as np
import mujoco
import mediapy as media
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Mujoco and custom environments
import gymnasium as gym
import ext_ctrl_envs


# Create the env
env_id = "NonnonimalCartpole"
env = gym.make(env_id, render_mode="human")
env.reset()
env.render()

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

done = False
while not done:
    env.step(action=[0])