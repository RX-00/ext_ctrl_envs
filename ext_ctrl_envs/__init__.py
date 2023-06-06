from gymnasium.envs.registration import register

register(id=f"NonnonimalCartpole",
         entry_point="ext_ctrl_envs.envs.cartpole.nonnominal_cartpole:NonnominalCartpoleEnv",
         max_episode_steps=500,)

register(id=f"NominalCartpole",
         entry_point="ext_ctrl_envs.envs.cartpole.nominal_cartpole:NominalCartpoleEnv",
         max_episode_steps=500,)


__version__ = "0.0.1"
