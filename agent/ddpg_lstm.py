import os
import logging
import random
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input, LSTM
from keras.layers.merge import add
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from utils import orn_uhlen, discount
from ddpg import DDPG, ActorNetwork, CriticNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
BUFFER_SIZE = 100000
BATCH_SIZE = 1
GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters
LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic
EXPLORE = 10000.
MAX_STEPS = 100000


class DDPG_LSTM(DDPG):
    def train(self, episodes):
        logging.info('Playing pretraining episodes (without training)')
        [self.play_episode(train=False) for _ in range(1)]
        logging.info('Playing training episodes')
        return [self.play_episode() for _ in range(episodes)]

    def play_episode(self, train=True):
        # TODO
        state = self.env.reset(train)
     
        e_states = []
        e_actions = []
        e_rewards = []
        e_newstates = []
        e_dones = []

        for step_count in range(MAX_STEPS):
            self.loss = 0 
            self.epsilon -= 1.0 / EXPLORE
            
            actions = self.actor.model.predict(state.reshape(1, 1, state.shape[0])).ravel()
            actions = self.noise_actions(actions, self.epsilon)

            new_state, reward, done = self.env.step(actions)

            e_states.append(state)
            e_actions.append(actions)
            e_rewards.append(reward)
            e_newstates.append(new_state)
            e_dones.append(done)
            

            state = new_state
        
            if not step_count % 50:
                logging.info("Step: {0} Reward: {1} Actions: {2}".format(step_count, reward, actions))
        
            if done:
                self.save_weights()
                break

        # for st, act, rew, nst, d in zip(e_states, e_actions, discount(e_rewards).tolist(), e_newstates, e_dones):
        self.buff.add([e_states, e_actions, e_rewards, e_newstates, e_dones])
        
        if train:
            self.loss = self._train_episode()

        logging.info("Total reward for episode: {0}".format(discount(e_rewards).sum()))
        return discount(e_rewards).sum()

    def predict(self, state):
        pass
        # TODO
        # return self.actor.model.predict(state.reshape(1, state.shape[0])).ravel()

    def _train_episode(self):
        # TODO
        episodes = self.buff.get_batch(10)
        for states, actions, reward, new_states, done in episodes:
            states = np.asarray(states)
            actions = np.asarray(actions)
            reward = np.asarray(reward)
            new_states = np.asarray(new_states)
            done = np.asarray(done)

            self.reset_states()

            for i in range(states.shape[0]):
                prediction = self.actor.target_model.predict(new_states[np.newaxis, np.newaxis, i])
                target_q_values = self.critic.target_model.predict([new_states[np.newaxis, np.newaxis, i], prediction[np.newaxis, :]])
                qval = np.asarray([[reward[i], reward[i]]]) if done[i] else  reward[i] + GAMMA*target_q_values
                self.loss += self.critic.model.train_on_batch([states[np.newaxis, np.newaxis, i], actions[np.newaxis, np.newaxis, i]], qval)
                a_for_grad = self.actor.model.predict(states[np.newaxis, np.newaxis, i])
                grads = self.critic.gradients(states[np.newaxis, np.newaxis, i], a_for_grad[np.newaxis, :])
                self.actor.train(states[np.newaxis, np.newaxis, i], grads.squeeze(axis=0))
                self.actor.target_train()
                self.critic.target_train()
        return self.loss

    def reset_states(self):
        self.actor.model.reset_states()
        self.critic.model.reset_states()


class ActorNetworkLSTM(ActorNetwork):
    def create_actor_network(self, state_size, actions_dim):
        logging.info("Build the actor model")
        S = Input(batch_shape=[BATCH_SIZE, 1, state_size])
        lstm = LSTM(HIDDEN1_UNITS, stateful=True)(S)
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(lstm)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        V = Dense(actions_dim, activation='tanh', kernel_initializer='random_normal')(h1)
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S


class CriticNetworkLSTM(CriticNetwork):
    def create_critic_network(self, state_size,actions_dim):
        logging.info("Build the critic model")
        S = Input(batch_shape=[BATCH_SIZE, 1, state_size])  
        A = Input(batch_shape=[BATCH_SIZE, 1, actions_dim], name='action2')
        lstm_s = LSTM(HIDDEN1_UNITS, stateful=True)(S)
        lstm_a = LSTM(HIDDEN1_UNITS, stateful=True)(A)
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(lstm_s)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(lstm_a) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = add([h1, a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(actions_dim,activation='linear')(h3)   
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class ReplayBuffer:
    # TODO
    def __init__(self, capacity):
        self.capacity = capacity
        self._buffer = deque()

    def _get_size(self):
        return len(self._buffer)

    size = property(_get_size)

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        batch_size = min(batch_size, self.size)
        return random.sample(self._buffer, batch_size)

    def add(self, episode):
        if self.size < self.capacity:
            self._buffer.append(episode)
        else:
            self._buffer.popleft()
            self._buffer.append(episode)

    def clear(self):
        self._buffer = deque()


def get_ddpg_lstm_agent(env):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    actor = ActorNetworkLSTM(sess, env.action_dim, env.state_dim, TAU, LRA)
    critic = CriticNetworkLSTM(sess, env.action_dim, env.state_dim, TAU, LRC)
    replay_buf = ReplayBuffer(BUFFER_SIZE)
    agent = DDPG_LSTM(env, actor, critic, replay_buf, sess)
    return agent

__all__ = ['DDPG_LSTM', 'get_ddpg_lstm_agent']
