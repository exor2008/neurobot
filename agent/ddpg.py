import os
import logging
import random
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import add
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from utils import orn_uhlen, discount
from baseagent import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic
EXPLORE = 10000.
MAX_STEPS = 100000


class WeightSerialization:
    def save_weights(self):
        self.actor.model.save_weights(os.path.join("model", "actormodel.h5"), overwrite=True)
        self.critic.model.save_weights(os.path.join("model", "criticmodel.h5"), overwrite=True)
        logging.info("Model saved")

    def load_weights(self, actor_file_name, critic_file_name):
        self.actor.model.load_weights(actor_file_name)
        self.critic.model.load_weights(critic_file_name)
        self.actor.target_model.load_weights(actor_file_name)
        self.critic.target_model.load_weights(critic_file_name)
        logging.info("Model loaded")


class DDPG(BaseAgent, WeightSerialization):
    def __init__(self, env, actor, critic, replay_buf, sess):
        super(DDPG, self).__init__(env)
        self.actor = actor
        self.critic = critic
        self.sess = sess
        self.buff = replay_buf
        self.epsilon = 1

    def train(self, episodes):
        logging.info('Playing pretraining episodes (without training)')
        [self.play_episode(train=False) for _ in range(5)]
        logging.info('Playing training episodes')
        return [self.play_episode() for _ in range(episodes)]

    def play_episode(self, train=True):
        state = self.env.reset(train)
     
        e_states = []
        e_actions = []
        e_rewards = []
        e_newstates = []
        e_dones = []

        for step_count in range(MAX_STEPS):
            self.loss = 0 
            self.epsilon -= 1.0 / EXPLORE
            
            actions = self.actor.model.predict(state.reshape(1, state.shape[0])).ravel()
            actions = self.noise_actions(actions, self.epsilon)

            new_state, reward, done = self.env.step(actions)

            e_states.append(state)
            e_actions.append(actions)
            e_rewards.append(reward)
            e_newstates.append(new_state)
            e_dones.append(done)
            
            if train:
                self.loss = self._train_step()

            state = new_state
        
            if not step_count % 50:
                logging.info("Step: {0} Reward: {1} Loss: {2} Actions: {3}".format(step_count, reward, self.loss, actions))
        
            if done:
                self.save_weights()
                break

        for st, act, rew, nst, d in zip(e_states, e_actions, discount(e_rewards).tolist(), e_newstates, e_dones):
            self.buff.add(st, act, rew, nst, d)

        logging.info("Total reward for episode: {0}".format(discount(e_rewards).sum()))
        return discount(e_rewards).sum()

    def predict(self, state):
        return self.actor.target_model.predict(state.reshape(1, state.shape[0])).ravel()

    def _train_step(self):
        batch = self.buff.get_batch(BATCH_SIZE)
        states = np.vstack(batch['state'].values)
        actions = np.vstack(batch['action'].values)
        new_states = np.vstack(batch['new_state'].values)
        reward = np.vstack(batch['reward'].values)
        done = np.vstack(batch['done'].values)

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])  

        qval = np.where(done == False, reward + GAMMA*target_q_values, reward)
   
        self.loss += self.critic.model.train_on_batch([states, actions], qval) 
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        return self.loss

    def noise_actions(self, actions, epsilon):
        noise = max(epsilon, 0) * orn_uhlen(actions,  0.0 , 0.60, 0.30)
        actions += noise
        return actions


class ActorNetwork:
    def __init__(self, sess, action_size, state_size, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate

        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau*actor_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, actions_dim):
        logging.info("Build the actor model")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        V = Dense(actions_dim, activation='tanh', kernel_initializer='random_normal')(h1)
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S


class CriticNetwork:
    def __init__(self, sess, action_size, state_size, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_size = action_size

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,actions_dim):
        logging.info("Build the critic model")
        S = Input(shape=[state_size])  
        A = Input(shape=[actions_dim], name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = add([h1, a1])    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(actions_dim,activation='linear')(h3)   
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class ReplayBuffer:
    def __init__(self, capacity, columns=None):
        self.capacity = capacity
        self.columns = columns
        self._buffer = deque()

    def _get_size(self):
        return len(self._buffer)

    size = property(_get_size)

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        batch_size = min(batch_size, self.size)
        batch = np.asarray(random.sample(self._buffer, batch_size))
        batch = pd.DataFrame(batch)
        if self.columns:
            batch.columns = self.columns
        return batch

    def add(self, state, actions, reward, new_state, done):
        if self.size < self.capacity:
            self._buffer.append([state, actions, reward, new_state, done])
        else:
            self._buffer.popleft()
            self._buffer.append([state, actions, reward, new_state, done])

    def clear(self):
        self._buffer = deque()


def get_ddpg_agent(env):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    actor = ActorNetwork(sess, env.action_dim, env.state_dim, TAU, LRA)
    critic = CriticNetwork(sess, env.action_dim, env.state_dim, TAU, LRC)
    replay_buf = ReplayBuffer(BUFFER_SIZE, ['state', 'action', 'reward', 'new_state', 'done'])
    agent = DDPG(env, actor, critic, replay_buf, sess)
    return agent

__all__ = ['DDPG', 'get_ddpg_agent']