from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
)


register(
    id="SpecialCartPole-v0",
    entry_point="CustomEnvs:SpecialCartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="SpecialMountainCar-v0",
    entry_point="CustomEnvs:SpecialMountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="SpecialAcrobot-v1",
    entry_point="CustomEnvs:SpecialAcrobotEnv",
    max_episode_steps=500,
    reward_threshold=-100.0,
)

register(
    id="SpecialPendulum-v1",
    entry_point="CustomEnvs:SpecialPendulumEnv",
    max_episode_steps=200,
)