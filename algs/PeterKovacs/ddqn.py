from __future__ import print_function

import os

import tensorflow as tf
import gym

import config as cfg
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise


class DDQN:
    def __init__(self, sess, env, data_folder):
        self.sess = sess
        self.env = env

        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        self.actor = ActorNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
        self.critic = CriticNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)
        self.buff = ReplayBuffer(cfg.BUFFER_SIZE)
        self.exploration = OUNoise(act_dim)

        self.saver = tf.train.Saver()
        self.data_folder = data_folder
        self.load()

    def train(self):
        for ep in range(cfg.EPISODES):
            s, reward, done = self.env.reset(), 0, False
            self.exploration.reset()

            for t in range(cfg.STEPS):
                self.env.render()

                # execute step
                a = self.actor.predict([s]) + self.exploration.noise()
                ns, r, done, info = self.env.step(a[0])
                self.buff.add(s, a[0], r, ns, done)

                # sample minibatch
                batch = self.buff.getBatch(cfg.BATCH_SIZE)
                states, actions, rewards, new_states, dones = zip(*batch)

                # set target
                target_q = self.critic.target_predict(new_states, self.actor.target_predict(new_states))
                y = [rewards[i] + (cfg.GAMMA * target_q[i] if not dones[i] else 0) for i in range(len(batch))]

                # update critic
                self.critic.train(y, states, actions)

                # update actor
                grads = self.critic.gradients(states, self.actor.predict(states))
                self.actor.train(states, grads)

                # update the target networks
                self.actor.target_train()
                self.critic.target_train()

                # move to next state
                s = ns
                reward += r

            if ep % cfg.SAVE_EVERY_EPISODES == 0:
                self.save()

            print("%3d  Reward = %+7.0f  " % (ep, reward))

    def run(self):
        for ep in range(cfg.EPISODES):
            s = self.env.reset()
            reward = 0

            for t in range(cfg.STEPS):
                self.env.render()

                a = self.actor.predict([s])
                s, r, _, _ = self.env.step(a[0])
                reward += r

            print("%3d  Reward = %+7.0f  " % (ep, reward))

    def save(self):
        print("Saving...")
        self.saver.save(self.sess, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.saver.restore(self.sess, self.model_path)
            print("Successfully loaded:", self.model_path)
        else:
            print("Could not find old network weights for ", self.model_path)

    @property
    def model_path(self):
        name = "%s.%s" % (self.__class__.__name__, self.env.spec.id)
        return os.path.join(self.data_folder, name + ".ckpt")
