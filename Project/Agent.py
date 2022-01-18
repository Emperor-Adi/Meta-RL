import numpy as np
import tensorflow as tf
from tensorflow import keras
from Memory import *
from tensorflow.python.framework.ops import disable_eager_execution

class Agent:
    def __init__(self, action_n, state_dim, batch_size, GAMMA=0.99, \
        GAE_LAMBDA=0.95, CLIP_LOSS_RATIO=0.1, ENTROPY_LOSS_RATIO=0.001, ALPHA=0.9):
        self.action_n = action_n
        self.state_dim = state_dim
        self.BATCH_SIZE = batch_size
        self.GAMMA = GAMMA
        self.GAE_LAMBDA = GAE_LAMBDA
        self.CLIP_LOSS_RATIO = CLIP_LOSS_RATIO
        self.ENTROPY_LOSS_RATIO = ENTROPY_LOSS_RATIO
        self.ALPHA = ALPHA
        disable_eager_execution()
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


    def ppo_loss(self, advantage, old_prediction):
        """The PPO custom loss.
        To add stability to policy updates.
        params:
            :advantage: advantage, needed to process algorithm
            :old_prediction: prediction from "old network", needed to process algorithm
        """
        def loss_fn(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            ratio = prob / (old_prob + 1e-10)
            clip_ratio = keras.backend.clip(ratio, min_value=(1-self.CLIP_LOSS_RATIO), max_value=(1+self.CLIP_LOSS_RATIO))
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            entropy_loss = (prob * keras.backend.log(prob + 1e-10))
            ppo_loss = -keras.backend.mean(keras.backend.minimum(surrogate1,surrogate2) + self.ENTROPY_LOSS_RATIO * entropy_loss)
            return ppo_loss
        return loss_fn


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
        actor_network = keras.Model(inputs = [state, advantage, old_prediction], outputs = policy)
        actor_network.compile(
            optimizer='Adam',
            loss = self.ppo_loss(advantage=advantage,old_prediction=old_prediction)
            )
        #actor_network.summary()
        return actor_network


    def choose_action(self,state):
        assert isinstance(state,np.ndarray)
        state = np.reshape(state,[-1,self.state_dim[0]])
        prob = self.actor_network.predict_on_batch([state,self.dummy_advantage, self.dummy_old_prediciton]).flatten()
        action = np.random.choice(self.action_n,p=prob)
        return action


    def get_v(self, state):
        s = np.reshape(state,(-1, self.state_dim[0]))
        v = self.critic_network.predict_on_batch(s)
        return v

    def make_gae(self):
        """Generates GAE-Generalized Advantage Estimation type rewards and pushes them into memory object
            #delta = r + gamma * V(s') * mask - V(s)  | aka advantage 
            #gae = delta + gamma * lambda * mask * gae | moving average smoothing 
            #return(s,a) = gae + V(s)  | add value of state back to it. 
        """
        gae = 0
        mask = 0
        for i in range(self.memory.sample_count):
            mask = 0 if self.memory.batch_done[i] else 1
            v = self.get_v(self.memory.batch_s[i])
            v_ = self.get_v(self.memory.batch_s_[i])
            delta = (self.memory.batch_r[i] + v_ * self.GAMMA * mask - v)
            gae = (delta + self.GAMMA * self.GAE_LAMBDA * mask * gae)
            self.memory.batch_gae_r.append(gae + v)
        self.memory.GAE_CALCULATED_Q = True


    def get_old_prediction(self, states):
        return_batch = []
        for state in states:
            state = np.reshape(state, (-1, self.state_dim[0]))
            return_batch.append(self.actor_old_network.predict_on_batch \
                ([state, self.dummy_advantage, self.dummy_old_prediciton])[0])
        return np.array(return_batch)


    def update_target_network(self):
        alpha = self.ALPHA
        actor_weights = np.array(self.actor_network.get_weights(), dtype=object)
        actor_target_weights = np.array(self.actor_old_network.get_weights(), dtype=object)
        new_weights = alpha*actor_weights + (1-alpha)*actor_target_weights
        self.actor_old_network.set_weights(new_weights)


    def train_network(self):
        if not self.memory.GAE_CALCULATED_Q:
            self.make_gae()
        
        states,actions,gae_rewards = self.memory.get_batch(self.BATCH_SIZE)
        batch_s = np.vstack(states)
        batch_a = np.vstack(actions)
        batch_gae_r = np.vstack(gae_rewards)

        batch_v = self.get_v(batch_s)
        batch_adv = batch_gae_r - batch_v
        batch_adv = keras.utils.normalize(batch_adv)
        
        batch_old_preds = self.get_old_prediction(batch_s)
        batch_old_preds = keras.utils.normalize(batch_old_preds)

        batch_a_final = np.zeros(shape=(len(batch_a),self.action_n))
        batch_a_final[:, batch_a.flatten()] = 1

        self.actor_network.fit(x=[batch_s, batch_adv, batch_old_preds], \
            y=batch_a_final, verbose=0)
        self.critic_network.fit(x=batch_s, y=batch_gae_r, verbose=0)
        self.update_target_network()