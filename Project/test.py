import gym
import csv
import os
from .agent import Agent
from .config import extern
from .GymWrapper.GymWrap import GymWrap


class Tester:
    """
    Class to test an agent on a gym environment.
    """
    @extern
    def __init__(self,ID,ENV_NAME="",MODEL_PATH="") -> None:
        self.env = gym.make(ENV_NAME)
        self.agent = Agent(self.env.action_space.n,self.env.observation_space.shape)
        self.samples_filled = 0
        self.env_name = ENV_NAME
        self.model = MODEL_PATH
        self.ID = ID


    def log(self) -> None:
        file_name = '../Logs/test/'+self.env_name+self.ID+'.csv'
        if not os.path.isfile(file_name):
            with open(file_name,'w') as csvfile:
                header = csv.writer(csvfile)
                fields = ['Environment Name','Environnment Type','Model',\
                    'Episode','Episodic Reward']
                header.writerow(fields)
        self.logfile = open(file_name,'a+')
        self.logger = csv.writer(file_name,lineterminator="\n")
        
    

    def cleanup(self) -> None:
        self.env.close()
        self.logfile.close()
    

    @extern
    def wrap(self,ENV_TYPE="") -> None:
        self.env = GymWrap(self.env,self.env_name,ENV_TYPE)


    @extern
    def test(self,TEST_ITERATIONS=50) -> None:
        """
        Test the trained agent on a modified Gym environment
        """

        self.log()
        self.wrap()

        self.agent.load_model(self.model)
        log_row = [self.env_name,self.env.env_type,self.model,None,None]
        for episode in range(TEST_ITERATIONS):
            episode_reward = 0
            state  = self.env.reset()
            while True:
                action = self.agent.choose_action(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if done:
                    break
            log_row[3],log_row[4] = episode,episode_reward
            self.logger.writerow(log_row)

        self.cleanup()
