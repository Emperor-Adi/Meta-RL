import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
# from tensorflow.keras import layers, models
from .Memory import Memory
import time



class Agent:
    def __init__(self, action, state_shape, batch_size, GAMMA=0.99, \
        GAE_LAMBDA=0.95, CLIP_LOSS_RATIO=0.1, ENTROPY_LOSS_RATIO=0.001, ALPHA=0.9) -> None:
        """
        Initializes the agent with the given parameters and builds the A2C networks.
        """
        # Agent parameters
        self.action = action
        self.state_shape = state_shape
        self.batch_size = batch_size
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.clip_loss_ratio = CLIP_LOSS_RATIO
        self.entropy_loss_ratio = ENTROPY_LOSS_RATIO
        self.alpha = ALPHA
        # Agent networks
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights())
        # Agent variables
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, self.action))
        # Agent memory
        self.memory = Memory()


    def _build_critic_network(self) -> models.Model:
        """
        Builds and returns a compiled keras model for the critic.
        The critic is a scalar prediction on the state value(output) given a state(input)
        Loss is mean-squared error
        """
        state = layers.Input(shape=self.state_shape,name='state_input')
        dense1 = layers.Dense(32,activation='relu',name='dense1')(state)
        dense2 = layers.Dense(32,activation='relu',name='dense2')(dense1)
        outputs = layers.Dense(1, name="actor_output_layer")(dense2)
        critic_network = models.Model(inputs=state, outputs=outputs)
        critic_network.compile(optimizer='Adam',loss = 'mean_squared_error')
        critic_network.summary()
        time.sleep(0.5)
        return critic_network


    def custom_loss(self, advantage, old_prediction) -> tf.function:
        """
        Computes the custom PPO loss
        Adds stability to policy updates
        params:
            :advantage: Advantage vector
            :old_prediction: Old action prediction
        returns:
            :loss: Loss function
        """
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            ratio = prob / (old_prob + 1e-10)
            ratio_clipped = tf.clip_by_value(ratio, 1-self.clip_loss_ratio, 1+self.clip_loss_ratio)
            surrogate1 = ratio * advantage
            surrogate2 = ratio_clipped * advantage
            entropy_loss = prob * tf.math.log(prob + 1e-10)
            ppo_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2) + self.entropy_loss_ratio * entropy_loss)
            return ppo_loss
        return loss
    

    def _build_actor_network(self)->models.Model:
        """
        Builds and returns a compiled keras model for the actor
        There are 3 inputs, out of which only the state is for the pass through the NN
        The advantage and old_prediction are exclusively for the custom loss fn (PPO)
        """
        state = layers.Input(shape=self.state_shape,name='state_input')
        advantage = layers.Input(shape=(1,),name='advantage_input')
        old_prediction = layers.Input(shape=(self.action,),name='old_prediction_input')
        rnn_in = tf.expand_dims(state, axis=0)
        lstm = layers.LSTM(24,activation='relu',name='lstm')(rnn_in)
        dense1 = layers.Dense(32,activation='relu',name='dense1')(lstm)
        dense2 = layers.Dense(32,activation='relu',name='dense2')(dense1)
        policy = layers.Dense(self.action, activation='softmax',name='actor_output_layer')(dense2)
        actor_network = models.Model(inputs=[state, advantage, old_prediction], outputs=policy)
        actor_network.compile(
            optimizer='Adam',
            loss = self.custom_loss(advantage, old_prediction)
            )
        actor_network.summary()
        return actor_network
   

    def choose_action(self,state) -> int:
        assert isinstance(state,np.ndarray)
        state = np.reshape(state,[-1,self.state_shape[0]])
        prob = self.actor_network.predict_on_batch([state,self.dummy_advantage, self.dummy_old_prediciton]).flatten()
        action = np.random.choice(self.action,p=prob)
        return action
    

    def get_old_prediction(self, states)->np.array:
        ret_batch = []
        for state in states:
            state = np.reshape(state, (-1, self.state_shape[0]))
            ret_batch.append(self.actor_old_network.predict_on_batch \
                ([state,self.dummy_advantage,self.dummy_old_prediciton])[0]
            )
        return np.array(ret_batch)
    
    
    def get_val(self,state) -> np.array:
        state = np.reshape(state,(-1,self.state_shape[0]))
        val = self.critic_network.predict_on_batch(state)
        return val


    def make_gae(self) -> None:
        """
        Generate Generalized Advantage Estimation(GAE) vector and store in memory
        delta = r + gamma * V(s') * mask - V(s) | Advantage
        gae = delta + gamma * lambda * mask * gae | Moving average smoothing
        returns gae + V(s)
        """
        gae = 0
        mask = 0
        for i in reversed(range(self.memory.size())):
            mask = 0 if self.memory.batch_done[i] else 1
            val = self.get_val(self.memory.batch_s[i])
            val_next = self.get_val(self.memory.batch_s_[i])
            delta = self.memory.batch_r[i] + self.gamma * val_next * mask - val
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            self.memory.batch_gae_r.append(gae + val)
        self.memory.GAE_CALCULATED_Q = True

    
    def store_transition(self,s,a,s_,r,done) -> None:
        self.memory.store(s, a, s_, r ,done)


    def normalize(x, axis=-1, order=2) -> np.array:
        l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
        l2[l2 == 0] = 1
        return x / np.expand_dims(l2, axis)


    def update_old_network(self) -> None:
        actor_weights = np.array(self.actor_network.get_weights())
        actor_old_weights = np.array(self.actor_old_network.get_weights())
        new_weights = self.alpha * actor_weights + (1 - self.alpha) * actor_old_weights
        self.actor_old_network.set_weights(new_weights)


    def train_network(self) -> None:
        if not self.memory.GAE_CALCULATED_Q:
            self.make_gae()
        s,a,s_,r,gae_r,done = self.memory.get_batch(self.batch_size)

        batch_s = np.vstack(s)
        batch_a = np.vstack(a)
        batch_gae_r = np.vstack(gae_r)
        batch_v = self.get_val(batch_s)
        batch_adv = self.normalize(batch_gae_r - batch_v)
        batch_old_preds = self.get_old_prediction(batch_s)

        batch_a_final = np.zeros((len(batch_a),self.action))
        batch_a_final[:,batch_a.flatten()] = 1

        self.actor_network.fit(
            x=[batch_s, batch_adv, batch_old_preds],
            y=batch_a_final,
            batch_size=self.batch_size
        )
        self.critic_network.fit(
            x=batch_s,
            y=batch_gae_r,
            epochs=1,
            batch_size=self.batch_size
        )
        self.update_old_network()
