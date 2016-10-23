from __future__ import print_function

import os

import tensorflow as tf
import config as cfg
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise


class DDQN:
    def __init__(self, sess, env_id, obs_dim, act_dim, data_folder=None):
        self.sess = sess
        self.env_id = env_id

        self.actor = ActorNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
        self.critic = CriticNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)
        self.buff = ReplayBuffer(cfg.BUFFER_SIZE)
        self.exploration = OUNoise(act_dim)

        self.saver = tf.train.Saver()
        self.data_folder = data_folder
        self.load()

    def train(self, env, episodes, steps, save_every_episodes):
        for ep in range(episodes):
            s, reward, done = env.reset(), 0, False
            self.exploration.reset()

            for t in range(steps):
                env.render()

                # execute step
                a = self.actor.predict([s]) + self.exploration.noise()
                ns, r, done, info = env.step(a[0])
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

            if ep % save_every_episodes == 0:
                self.save()

            print("%3d  Reward = %+7.0f  " % (ep, reward))

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
        if self.model_path is None:
            return
        if os.path.exists(self.model_path):
            self.saver.restore(self.sess, self.model_path)
            print("Successfully loaded:", self.model_path)
        else:
            print("Could not find old network weights for ", self.model_path)

    @property
    def model_path(self):
        if self.data_folder is None:
            return None
        name = "%s.%s" % (self.__class__.__name__, self.env_id)
        return os.path.join(self.data_folder, name + ".ckpt")
