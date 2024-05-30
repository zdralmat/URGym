from gymnasium.envs.registration import register

# Register the environment
register(
    id='ur5/box-v0',
    entry_point='ur5.envs.env_box:BoxManipulation',
    max_episode_steps=200,
    kwargs=dict(button_touch_mode='any')
)


register(
    id='ur5/cubes_push-v0',
    entry_point='ur5.envs.env_cubes_push_joint:CubesPush',
    max_episode_steps=50,
)


# Register the environment
register(
    id='ur5/cubes_grasp_joint-v0',
    entry_point='ur5.envs.env_cubes_grasp_joint:CubesGrasp',
    max_episode_steps=10,
)