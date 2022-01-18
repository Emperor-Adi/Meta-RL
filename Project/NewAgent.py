import numpy as np
import tensorflow as tf
from tensorflow import keras
from Memory import *

class Agent:
    def __init__(self, action_n, state_dim, batch_size, GAMMA=0.99, \
        GAE_LAMBDA=0.95, CLIPPING_LOSS_RATIO=0.1, ENTROPY_LOSS_RATIO=0.001, TARGET_UPDATE_ALPHA=0.9):
        self.action_n = action_n
        self.state_dim = state_dim
        self.BATCH_SIZE = batch_size
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
    
    def _build_critic_network(self):
        """builds and returns a compiled keras.model for the critic.
        The critic is a simple scalar prediction on the state value(output) given an state(input)
        Loss is simply mean-squared error
        """
        state = keras.layers.Input(shape=self.state_dim,name='state_input')
        dense = keras.layers.Dense(32,activation='relu',name='dense1')(state)
        dense = keras.layers.Dense(32,activation='relu',name='dense2')(dense)
        V = keras.layers.Dense(1, name="actor_output_layer")(dense)
        critic_network = keras.Model(inputs=state, outputs=V)
        critic_network.compile(optimizer='Adam',loss = 'mean_squared_error')
        #critic_network.summary()
        return critic_network

    def _build_actor_network(self):
        """builds and returns a compiled keras.model for the actor.
        There are 3 inputs. Only the state is for the pass though the neural net.
        The other two inputs are exclusivly used for the custom loss function (ppo_loss).
        """

        state = keras.layers.Input(shape=self.state_dim,name='state_input')
        advantage = keras.layers.Input(shape=(1,),name='advantage_input')
        old_prediction = keras.layers.Input(shape=(self.action_n,),name='old_prediction_input')
        rnn_in = tf.expand_dims(state, [0])
        lstm = keras.layers.LSTM(24,activation='relu')(rnn_in)
        dense = keras.layers.Dense(32,activation='relu',name='dense1')(lstm)
        dense = keras.layers.Dense(32,activation='relu',name='dense2')(dense)
        policy = keras.layers.Dense(self.action_n, activation="softmax", name="actor_output_layer")(dense)
        actor_network = keras.Model(inputs = [state,advantage,old_prediction], outputs = policy)
        actor_network.compile(
            optimizer='Adam',
            loss = self.ppo_loss(advantage=advantage,old_prediction=old_prediction)
            )
        #actor_network.summary()
        return actor_network    


    def get_v(self,state):
        s = np.reshape(state,(-1, self.state_dim[0]))
        v = self.critic_network.predict_on_batch(s)
        return v

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

