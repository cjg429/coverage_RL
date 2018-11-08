from gym.envs.registration import register

register(
    id='Coverage-v0',
    entry_point='gym_coverage.envs:CoverageEnv',
)
