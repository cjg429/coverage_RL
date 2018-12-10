from gym.envs.registration import register

register(
    id='IRL-v0',
    entry_point='gym_IRL.envs:IRLEnv',
)