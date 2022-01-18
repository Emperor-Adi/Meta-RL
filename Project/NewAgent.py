import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from Memory import *

class Agent:
    def __init__(self, action_n, state_dim, batch_size, GAMMA=0.99, \
        GAE_LAMBDA=0.95, CLIPPING_LOSS_RATIO=0.1, ENTROPY_LOSS_RATIO=0.001, TARGET_UPDATE_ALPHA=0.9):
        self.action_n = action_n
        self.state_dim = state_dim
        self.TRAINING_BATCH_SIZE = batch_size
        self.GAMMA = GAMMA
        self.GAE_LAMBDA = GAE_LAMBDA
        self.CLIPPING_LOSS_RATIO = CLIPPING_LOSS_RATIO
        self.ENTROPY_LOSS_RATIO = ENTROPY_LOSS_RATIO
        self.TARGET_UPDATE_ALPHA = TARGET_UPDATE_ALPHA
        
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights())
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, self.action_n))
        self.memory = Memory()
    
    def make_gae(self):
        """Generates GAE-Generalized advantage estimation type rewards and pushes them into memory object
            #delta = r + gamma * V(s') * mask - V(s)  |aka advantage
            #gae = delta + gamma * lambda * mask * gae |moving average smoothing
            #return(s,a) = gae + V(s)  |add value of state back to it.
        """
        gae = 0
        mask = 0
        for i in reversed(range(self.memory.sample_count)):
            mask = 0 if self.memory.batch_done[i] else 1
            v = self.get_v(self.memory.batch_s[i])
            delta = self.memory.batch_r[i] + self.GAMMA * self.get_v(self.memory.batch_s_[i]) * mask - v
            gae = delta + self.GAMMA *  self.GAE_LAMBDA * mask * gae
            self.memory.batch_gae_r.append(gae+v)
        self.memory.batch_gae_r.reverse()
        self.memory.GAE_CALCULATED_Q = True