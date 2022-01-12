from gym.envs.classic_control  import CartPoleEnv,MountainCarEnv,AcrobotEnv,PendulumEnv
# from gym.envs.mujoco import HalfCheetahEnv,half_cheetah_v3
from BoxScript import BoxSizes


class SpecialEnv(CartPoleEnv):
    def __init__(self, env_version):
        super().__init__()
        self.env_version = env_version
        err_msg = "Environment Version Parameter Error"
        assert self.env_version in ("Deterministic","Random","Extreme"), err_msg
        self.implversion()

    def reset(self):
        self.implversion()
        return super().reset()
    
    def implversion(self):
        self.force_mag = BoxSizes["CartPole-v1"][self.env_version][0]
        self.length= BoxSizes["CartPole-v1"][self.env_version][1]
        self.masspole= BoxSizes["CartPole-v1"][self.env_version][2]


class SpecialEnv(MountainCarEnv):
    def __init__(self, env_version, goal_velocity=0):
        super().__init__(goal_velocity=goal_velocity)
        self.env_version = env_version
        err_msg = "Environment Version Parameter Error"
        assert self.env_version in ("Deterministic","Random","Extreme"), err_msg
        self.implversion()

    def reset(self):
        self.implversion()
        return super().reset()
    
    def implversion(self):
        self.force = BoxSizes["MountainCar-v0"][self.env_version][0]
        self.gravity = BoxSizes["MountainCar-v0"][self.env_version][1]


class SpecialEnv(AcrobotEnv):
    def __init__(self, env_version):
        super().__init__()
        self.env_version = env_version
        err_msg = "Environment Version Parameter Error"
        assert self.env_version in ("Deterministic","Random","Extreme"), err_msg
        self.implversion()
    
    def reset(self):
        self.implversion()
        return super().reset()
    
    def implversion(self):
        self.LINK_LENGTH_1 = self.LINK_LENGTH_2 = BoxSizes["Acrobot-v1"][self.env_version][0]
        self.LINK_MASS_1 = self.LINK_MASS_2 = BoxSizes["Acrobot-v1"][self.env_version][1]
        self.LINK_MOI = BoxSizes["Acrobot-v1"][self.env_version][2]


class SpecialEnv(PendulumEnv):
    def __init__(self, env_version):
        super().__init__(g=10)
        self.env_version = env_version
        err_msg = "Environment Version Parameter Error"
        assert self.env_version in ("Deterministic","Random","Extreme"), err_msg
        self.implversion()
    
    def reset(self):
        self.implversion()
        return super().reset()

    def implversion(self):
        self.l = BoxSizes["Pendulum-v0"][self.env_version][0]
        self.m = BoxSizes["Pendulum-v0"][self.env_version][1]
