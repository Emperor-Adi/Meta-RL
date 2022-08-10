
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models, optimizers
from Memory import Memory
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
    

    