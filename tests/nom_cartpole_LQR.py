'''

Program of an LQR controller applied to a mujoco simulation of an
nominal inverted pendulum on a cart system.


'''

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

# getting riccati solver
from scipy import linalg



'''
Mujoco Gymnasium Environment Setup
'''
# Create the env
env_id = "NominalCartpole"
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
theta_positions = [state[1]]

# for if the system is in a termination or truncation state
done = False


'''
LQR Controller (hand written method)
'''
# constants and properties of the system
# NOTE: be sure to make sure these are in line with the .xml mujoco model
g = 9.81
lp = 1.0
mp = 0.1
mc = 1.0

# state transition matrix
a1 = (-12*mp*g) / (13*mc+mp)
a2 = (12*(mp*g + mc*g)) / (lp*(13*mc + mp))
A = np.array([[0, 1, 0,  0],
              [0, 0, a1, 0],
              [0, 0, 0,  1],
              [0, 0, a2, 0]])

# control transition matrix
b1 = 13 / (13*mc + mp)
b2 = -12/ (lp*(13*mc + mp))
B = np.array([[0 ],
              [b1],
              [0 ],
              [b2]])

R = np.eye(1, dtype=int) * 10     # choose R (weight for input), we want input to be min.
Q = np.array([[10,  0,  0,  0  ],
              [ 0,  1,  0,  0  ],
              [ 0,  0, 10,  0  ],
              [ 0,  0,  0,  1  ]])     # choose Q (weighted for cart pos and pendulum angle)

# Solves the continuous-time algebraic Riccati equation (CARE).
P = linalg.solve_continuous_are(A, B, Q, R)

# Calculate optimal controller gain
K = np.dot(np.linalg.inv(R),
           np.dot(B.T, P))
#print(K)

def apply_ctrlr(K, x):
    u = -np.dot(K, x)
    return u

# storing ctrl inputs
us = [np.array(0)]



'''
LQR Controller (mjData method)

Here we use the desired position of the system (state vector = 0 vector) as
the setpoint to linearize around to get our state transition matrix A = df/dx,
where f is some nonlinear dynamics.

We use inverse dynamics to find the best control u to linearize around to find
the control transition matrix B = df/du
'''
# set sys model to init_state
mujoco.mj_resetDataKeyframe(env.unwrapped.model, env.unwrapped.data, 0)
# we use mj_forward (forward dynamics function) to find the acceleration given
# the state and all the forces in the system
mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
env.unwrapped.data.qacc = 0 # Asset that there's no acceleration
# The inverse dynamics function takes accel as input and compute the forces
# required to create the acceleration. Uniquely, MuJoCo's fast inverse dyn.
# takes into accoun all constraint, including contacts
mujoco.mj_inverse(env.unwrapped.model, env.unwrapped.data)
# NOTE: make sure the required forces are achievable by your actuators before
#       continuing with the LQR controller process
#print(env.unwrapped.data.qfrc_inverse)

# Save the position and control setpoints to linearize around
qpos0 = env.unwrapped.data.qpos.copy()
qfrc0 = env.unwrapped.data.qfrc_inverse.copy()

# Finding actuator values that can create the desired forces
# for motor actuators (which we use) we can mulitple the control setpoints by
# the pseudo-inverse of the actuation moment arm
# NOTE: more elaborate actuators would require finite-differencing to recover
#       d qfrc_actuator / d u
print(qpos0)
print(qfrc0)
print(env.unwrapped.data.actuator_moment)

#ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(env.unwrapped.data.actuator_moment)
#ctrl0 = ctrl0.flatten() # save the ctrl setpoint
ctrl0 = 0.


# Choosing R
nu = env.unwrapped.model.nu # Alias for the number of actuators
# NOTE: Why do they use the number of actuators as the dimensions for R
#       and number of DoFs as the dimensions for Q????
#       -> Because the dimension of R represents the control cost
#          and Q represents the state of the system that we can observe,
#          hence the system (state) cost
R = np.eye(nu)

# Choosing Q
nv = env.unwrapped.model.nv # Alias for number of DoFs
# NOTE: this wasn't used
# To determine Q we'll be constructing it as a sum of two terms
#   term 1: a balancing cost that will keep the CoM over the cart
#           described by kinematic Jacobians which map b/w joint
#           space and global Cartesian positions (computed analytically)
#   term 2: a cost for joints moving away from their initial config

Q = np.array([[10,  0,   0,  0 ],
              [ 0,  1,   0,  0 ],
              [ 0,  0,  10,  0 ],
              [ 0,  0,   0,  1 ]])

# Computing gain matrix K
# Set the initial state and control.
mujoco.mj_resetData(env.unwrapped.model, env.unwrapped.data)
env.unwrapped.data.ctrl = ctrl0 # should be 0
env.unwrapped.data.qpos = qpos0 # should be 0

#
# Before we solve for the LQR controller, we need the A and B matrices.
# These are computed by MuJoCo's mjd_transitionFD function which computes
# them using efficient finite-difference derivatives, exploiting the
# configurable computation pipeline to avoid recomputing quantities which
# haven't changed.
#
A = np.zeros((2*nv, 2*nv))
B = np.zeros((2*nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(env.unwrapped.model, env.unwrapped.data,
                        epsilon, flg_centered, A, B, None, None)

print(A)

# Solve discrete Riccati equation.
P = linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

print(K)

def apply_ctrlr(K, x):
    u = -np.dot(K, x)
    return u


'''
Simulation

NOTE: step != timestep, please refer to the .xml file for the simulation timestep
      as that would effect the energy in the system.
'''
# reset sys model
env.reset()

#while not done: # while loop for training
for i in range(500): # for testing, 500 steps

    u = apply_ctrlr(K, state)

    #print("u: ", u)

    # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
    # These represent the next observation, the reward from the step,
    # if the episode is terminated, if the episode is truncated and
    # additional info from the step
    state, reward, terminated, truncated, info = env.step(action=u)

    # record data about system
    x_positions.append(state[0])
    theta_positions.append(state[1])
    us.append(u)

    if i == 100:
        # set environment to custom init_state (disturbed state)
        # TODO: figure out how to make this into a external pertebuation
        # mujoco.mj_resetDataKeyframe(env.unwrapped.model, env.unwrapped.data, 1)

        sys_qpos = env.unwrapped.data.qpos
        sys_qvel = env.unwrapped.data.qvel
        sys_qpos[0] = 1.8
        sys_qpos[1] = -0.5

        env.unwrapped.set_state(sys_qpos, sys_qvel)

    if i == 200:
        # Apply a force
        body_id = env.unwrapped.model.body("pole_1").id
        force  = np.array([-6.0, 0.0, 0.0]) # 5 N force in -x-dir
        torque = np.array([0.0, 0.0, 5.0]) # 5 Nm torque about z-axis
        pt_on_body  = np.array([0.0, 0.0, 0.0]) # Body origin

        #mujoco.mj_applyFT(env.unwrapped.model, env.unwrapped.data,
        #                  force, torque, pt_on_body,
        #                  body_id, data.qfrc_applied)

        env.unwrapped.data.xfrc_applied[body_id, :] = np.concatenate([force, torque])
    else:
        # Don't apply it continuously!
        env.unwrapped.data.xfrc_applied[2, :] = np.zeros(6)

    # End the episode when either truncated or terminated is true
    #  - truncated: The episode duration reaches max number of timesteps
    #  - terminated: Any of the state space values is no longer finite.
    done = terminated or truncated

env.close()

exit()

'''
Plotting data
'''
fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(x_positions)
axs[0].set_title('Cart position')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Position')

fig.suptitle('Cartpole x and theta pos', fontsize=16)

axs[1].plot(theta_positions)
axs[1].set_title('Pendulum angular position')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Radians')

axs[2].plot(us)
axs[2].set_title('Force applied on cart')
axs[2].set_xlabel('Time step')
axs[2].set_ylabel('Newtons')

print("A matrix: \n", A)
print("B matrix: \n", B)

plt.show()
