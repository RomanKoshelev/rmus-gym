from __future__ import print_function

import os

import tensorflow as tf
import config as cfg
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise
import numpy as np


class DDPG:
    def __init__(self, sess, env_id, obs_dim, obs_box, act_dim, act_box, data_folder, prefix=None):
        self.sess = sess
        self.prefix = prefix
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.obs_box = obs_box
        self.act_dim = act_dim
        self.act_box = act_box

        with tf.variable_scope(self.scope):
            with tf.variable_scope("actor"):
                self.actor = ActorNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
            with tf.variable_scope("critic"):
                self.critic = CriticNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)

        var_list = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.scope)
        self.sess.run(tf.initialize_variables(var_list))

        self.buff = ReplayBuffer(cfg.BUFFER_SIZE)
        self.exploration = OUNoise(act_dim, mu=0., sigma=0.2)

        self.saver = tf.train.Saver(var_list)
        self.data_folder = data_folder
        self.load()
        self.pint_summury()

    def train(self, env, episodes, steps, save_every_episodes):

        for ep in range(episodes):
            s, reward, done = env.reset(), 0, False
            self.exploration.reset()

            # calc noise rate
            nr_min = .2
            nr_max = 0.99
            nr_ep = episodes / 10
            nr_k = 1 - min(1, (float(ep) / nr_ep))
            nr = nr_min + nr_k * (nr_max - nr_min)

            for t in range(steps):
                env.render()

                # calc action
                a = self.actor.predict([s])  # type: np.ndarray
                n = self.exploration.noise()
                # a = np.clip(a, self.act_box[0], self.act_box[1])
                a = (1 - nr) * a + nr * n

                # todo: use act_box
                ae = a * 100.  # type: np.ndarray

                # print()
                # print(n)
                # print(a)

                # self.verify_actions(a)

                # execute step
                ns, r, done, info = env.step(ae[0])
                self.buff.add(s, a[0], r, ns, done)

                # sample minibatch
                batch = self.buff.getBatch(cfg.BATCH_SIZE)
                states, actions, rewards, new_states, dones = zip(*batch)

                # set target
                target_q = self.critic.target_predict(new_states, self.actor.target_predict(new_states))
                y = [rewards[i] + (cfg.GAMMA * target_q[i] if not dones[i] else [0]) for i in range(len(batch))]

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

            if ep > 0 and ep % save_every_episodes == 0:
                self.save()

            self.print_progress(ep, reward, nr)

    def run(self, env, episodes, steps):
        for ep in range(episodes):
            s = env.reset()
            reward = 0
            for t in range(steps):
                env.render()
                a = self.act(s)
                s, r, _, _ = env.step(a[0])
                reward += r
            print("%3d  Reward = %+7.0f  " % (ep, reward))

    def act(self, s):
        return self.actor.predict([s])

    def save(self):
        if self.model_path is None:
            return
        print("Saving...")
        self.saver.save(self.sess, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.saver.restore(self.sess, self.model_path)
            print("Successfully loaded:", self.model_path)
        else:
            print("Could not find old network weights for ", self.model_path)

    @property
    def scope(self):
        name = self.prefix + '_' if self.prefix is not None else ''
        name += "%s_%s" % (self.__class__.__name__, self.env_id)
        return name.replace('-', '_')

    @property
    def model_path(self):
        if self.data_folder is None:
            return None
        return os.path.join(self.data_folder, self.scope + ".ckpt")

    def pint_summury(self):
        print("\n==============================================================================")
        print("obs_dim: %d" % self.obs_dim)
        print("obs_box: %s" % self.obs_box[0])
        print("         %s" % self.obs_box[1])
        print("act_dim: %d" % self.act_dim)
        print("act_box: %s" % self.act_box[0])
        print("         %s" % self.act_box[1])
        print("==============================================================================\n")

    def print_progress(self, ep, reward, nr):
        print("%3d  Reward = %+7.0f   nr = %.2f" % (ep, reward, nr))

    def verify_actions(self, batch_actions):
        for action in batch_actions:
            res = action >= self.act_box[0]  # type: np.ndarray
            assert res.all(), "action=%f" % action
            res = action <= self.act_box[1]  # type: np.ndarray
            assert res.all(), "action=%f" % action
