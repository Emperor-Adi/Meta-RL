from gym import envs
from gym.envs.classic_control  import CartPoleEnv,MountainCarEnv,AcrobotEnv,PendulumEnv
import numpy as np


class SpecialCartPoleEnv(CartPoleEnv):
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
        if self.env_version == "Deterministic":
            self.force_mag = 10.0
            self.length = 0.5
            self.masspole = 0.1
        elif self.env_version == "Random":
            self.force_mag =  np.random.uniform(5.0,15.0)
            self.length = np.random.uniform(0.25,0.75)
            self.masspole = np.random.uniform(0.05,0.5)
        elif  self.env_version == "Extreme":
            self.force_mag = np.random.choice([np.random.uniform(1.0,5.0),np.random.uniform(15.0,20.0)])
            self.length = np.random.choice([np.random.uniform(0.05,0.25),np.random.uniform(0.75,1.0)])
            self.masspole = np.random.choice([np.random.uniform(0.01,0.05),np.random.uniform(0.5,1.0)])


class SpecialMountainCarEnv(MountainCarEnv):
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
        if self.env_version == "Deterministic":
            self.force = 0.001
            self.gravity = 0.0025
        elif self.env_version == "Random":
            self.force =  np.random.uniform(0.0005,0.005)
            self.gravity = np.random.uniform(0.001,0.005)
        elif  self.env_version == "Extreme":
            self.force = np.random.choice([np.random.uniform(0.0001,0.0005),np.random.uniform(0.005,0.01)])
            self.gravity = np.random.choice([np.random.uniform(0.0005,0.001),np.random.uniform(0.005,0.01)])


class SpecialAcrobotEnv(AcrobotEnv):
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
        if self.env_version == "Deterministic":
            self.LINK_LENGTH_1 = self.LINK_LENGTH_2 = 1
            self.LINK_MASS_1 = self.LINK_MASS_2 = 1
            self.LINK_MOI = 1
        elif self.env_version == "Random":
            self.LINK_LENGTH_1 = self.LINK_LENGTH_2 =  np.random.uniform(0.75,1.25)
            self.LINK_MASS_1 = self.LINK_MASS_2 = np.random.uniform(0.75,1.25)
            self.LINK_MOI = np.random.uniform(0.75,1.25)
        elif  self.env_version == "Extreme":
            self.LINK_LENGTH_1 = self.LINK_LENGTH_2 = np.random.choice([np.random.uniform(0.5,0.75),np.random.uniform(1.25,1.5)])
            self.LINK_MASS_1 = self.LINK_MASS_2 = np.random.choice([np.random.uniform(0.5,0.75),np.random.uniform(1.25,1.5)])
            self.LINK_MOI = np.random.choice([np.random.uniform(0.5,0.75),np.random.uniform(1.25,1.5)])


class SpecialPendulumEnv(PendulumEnv):
    def __init__(self, env_version, g=10):
        super().__init__(g=g)
        self.env_version = env_version
        err_msg = "Environment Version Parameter Error"
        assert self.env_version in ("Deterministic","Random","Extreme"), err_msg
        self.implversion()
    
    def reset(self):
        self.implversion()
        return super().reset()

    def implversion(self):
        if self.env_version == "Deterministic":
            self.l = 1
            self.m = 1
        elif self.env_version == "Random":
            self.l = np.random.uniform(0.75,1.25)
            self.m = np.random.uniform(0.75,1.25)
        elif  self.env_version == "Extreme":
            self.l = np.random.choice([np.random.uniform(0.5,0.75),np.random.uniform(1.25,1.5)])
            self.m = np.random.choice([np.random.uniform(0.5,0.75),np.random.uniform(1.25,1.5)])