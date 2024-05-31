from gymnasium.envs.registration import register

# Register the environment
register(
    id='URGym/Box-v0',
    entry_point='urgym.envs.env_box_v0:BoxManipulation',
    max_episode_steps=200,
    kwargs=dict(button_touch_mode='any')
)


register(
    id='URGym/CubesPush-v0',
    entry_point='urgym.envs.env_cubes_push_v0:CubesPush',
    max_episode_steps=50,
)


# Register the environment
register(
    id='URGym/CubesGrasp-v0',
    entry_point='urgym.envs.env_cubes_grasp_v0:CubesGrasp',
    max_episode_steps=20,
)