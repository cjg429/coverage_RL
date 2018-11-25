from gym.envs.registration import register

register(
    id='Coverage-v0',
    entry_point='gym_coverage.envs:CoverageEnv',
)

register(
    id='Coverage-v1',
    entry_point='gym_coverage.envs:Coverage_v1',
)

register(
    id='CoverageObs-v0',
    entry_point='gym_coverage.envs:CoverageObsEnv',
)

register(
    id='CoverageCount-v0',
    entry_point='gym_coverage.envs:CoverageCountEnv',
)

register(
    id='CoverageCar-v0',
    entry_point='gym_coverage.envs:CoverageCarEnv',
)
