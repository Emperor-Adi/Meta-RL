from gym import Wrapper
from BoxScript import BoxSizes,ENVS



class SpecialEnv(Wrapper):
    def __init__(self, env,  env_name, env_version):
        super().__init__(env)
        self.env_name = env_name
        self.env_version = env_version
        env_name_err = "Environment Not Recognized"
        assert self.env_name in ENVS, env_name_err
        env_ver_err = "Environment Version Parameter Error"
        assert self.env_version in ("Deterministic","Random","Extreme"), env_ver_err
        self.implversion()

    def reset(self):
        self.implversion()
        return super().reset()
    
    def implversion(self):
        if self.env_name == ENVS[0]:
            self.force_mag = BoxSizes[self.env_name][self.env_version][0]
            self.length= BoxSizes[self.env_name][self.env_version][1]
            self.masspole= BoxSizes[self.env_name][self.env_version][2]
        elif self.env_name == ENVS[1]:
            self.force = BoxSizes[self.env_name][self.env_version][0]
            self.gravity = BoxSizes[self.env_name][self.env_version][1]
        elif self.env_name == ENVS[2]:
            self.LINK_LENGTH_1 = self.LINK_LENGTH_2 = BoxSizes[self.env_name][self.env_version][0]
            self.LINK_MASS_1 = self.LINK_MASS_2 = BoxSizes[self.env_name][self.env_version][1]
            self.LINK_MOI = BoxSizes[self.env_name][self.env_version][2]
        elif self.env_name == ENVS[3]:
            self.l = BoxSizes[self.env_name][self.env_version][0]
            self.m = BoxSizes[self.env_name][self.env_version][1]
        elif self.env_name == ENVS[4]:
            pass
        elif self.env_name == ENVS[5]:
            pass
