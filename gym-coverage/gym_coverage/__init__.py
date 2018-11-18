from gym.envs.registration import register

register(
    id='Coverage-v0',
    entry_point='gym_coverage.envs:CoverageEnv',
)

register(
    id='CoverageObs-v0',
    entry_point='gym_coverage.envs:CoverageObsEnv',
)

register(
    id='CoverageCar-v0',
    entry_point='gym_coverage.envs:CoverageCarEnv',
)
