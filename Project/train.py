import numpy as np
import gym
from collections import deque
import csv
from time import time as timestamp
from .config import extern
from .agent import Agent
from .GymWrapper.GymWrap import GymWrap



class Trainer:
    """
    Class to train an agent on a gym environment.
    """
    @extern
    def __init__(self,ENV_NAME="") -> None:
        self.env = gym.make(ENV_NAME)
        self.agent = Agent(self.env.action_space.n,self.env.observation_space.shape)
        self.samples_filled = 0
        self.env_name = ENV_NAME


    def log(self) -> None:
        file_name = './Logs/train_data_'+str(timestamp)+'.csv'
        self.logfile = open(file_name,'w')
        self.logger = csv.writer(file_name,lineterminator="\n")
        self.logger.writerow(['Episode','Episodic Reward','Maximum Reward'])
        self.metadata = open(file_name+'.mtdt','w')
        self.metadata.writelines("Gym Environment: "+self.env_name+"\n")
    
    
    def cleanup(self) -> None:
        self.env.close()
        self.metadata.close()
        self.logfile.close()
        

    @extern
    def train(self,TRAIN_ITERATIONS=5000,EXPECTED_REWARD=500,MAX_EP_LEN=1000,\
        TRAJECTORY_BUFFER=64,BATCH_SIZE=32,WINDOW_SIZE=100,RENDER_FREQ=50) -> None:
        """
        Train the agent on the gym environment.
        """
        self.log()

        self.metadata.writelines("Expected Reward: "+str(EXPECTED_REWARD)+"\n")
        self.metadata.writelines("Batch Size: "+str(BATCH_SIZE)+"\n")
        self.metadata.writelines(
            "Trajectory Buffer Size: "+str(TRAJECTORY_BUFFER)+"\n")
        scores_window = deque(maxlen=WINDOW_SIZE)
        max_reward = -500
        for episode in range(TRAIN_ITERATIONS):
            state = self.env.reset()
            episode_reward = 0
            for step in range(MAX_EP_LEN):
                if step % RENDER_FREQ == 0:
                    self.env.render()
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.agent.store_transition(state, action, reward, next_state, done)
                self.samples_filled += 1
                if self.samples_filled == TRAJECTORY_BUFFER:
                    for _ in range(TRAJECTORY_BUFFER//BATCH_SIZE):
                        self.agent.train_network()
                    self.agent.memory.clear()
                    self.samples_filled = 0
                state = next_state
                if done:
                    break
            scores_window.append(episode_reward)
            max_reward  = max(max_reward,episode_reward)
            self.logger.writerow([episode+1,episode_reward,max_reward])
            if np.mean(scores_window)>=EXPECTED_REWARD:
                self.agent.actor_network.save_weights('./Models/actor_model_'+\
                    self.env_name+str(EXPECTED_REWARD)+'.h5')
                self.metadata.writelines("Solved in {} episodes".format(episode+1))
                break
        
        self.cleanup()
