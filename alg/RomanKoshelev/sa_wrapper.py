from __future__ import print_function

import numpy as np
import alg.PeterKovacs.config as cfg
from alg.PeterKovacs.ddgp import DDGP
from alg.PeterKovacs.ou_noise import OUNoise


class SAWrapper:
    def __init__(self, sess, env_id, obs_dim, act_dim, data_folder, prefix=None):
        self.sess = sess
        self.prefix = prefix
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ext_dim = 2
        self.data_folder = data_folder
        self.exploration = OUNoise(self.ext_dim)

    def run(self, env, episodes, steps):
        driver = DDGP(self.sess, self.env_id, self.obs_dim, self.ext_dim, self.data_folder, self.scope + '_driver')
        worker = DDGP(self.sess, self.env_id, self.obs_dim, self.act_dim, self.data_folder)
        for ep in range(episodes):
            s, reward, done = env.reset(), 0, False

            for t in range(steps):
                # set goal
                drv_s = s
                int_s = s[self.ext_dim:]
                drv_a = driver.act(drv_s)
                wrk_s = np.append(drv_a, int_s)

                # execute step
                wrk_a = worker.act(wrk_s)
                ns, r, done, info = env.step(wrk_a[0])
                s = ns
                reward += r

                # render
                env.metadata['extra'] = -drv_a[0]
                env.render()

            print("%3d  Reward = %+7.0f  " % (ep, reward))

    def train(self, env, episodes, steps, save_every_episodes):
        driver = DDGP(self.sess, self.env_id, self.obs_dim, self.ext_dim, self.data_folder, self.scope + '_driver')
        worker = DDGP(self.sess, self.env_id, self.obs_dim, self.act_dim, self.data_folder)

        for ep in range(episodes):
            s, reward, done = env.reset(), 0, False
            self.exploration.reset()

            noise_rate = max(0., 1. - float(ep) / episodes)

            for t in range(steps):
                # set goal
                drv_s = s
                int_s = s[self.ext_dim:]
                DA = 5.
                DN = DA / 2.
                drv_a = driver.act(drv_s)  # type: np.ndarray
                drv_a = np.clip(drv_a, [-DA, -DA], [DA, DA])
                drv_a += DN * noise_rate * self.exploration.noise()
                wrk_s = np.append(drv_a, int_s)

                # execute step
                wrk_a = worker.act(wrk_s)
                ns, r, done, info = env.step(wrk_a[0])
                drv_ns = ns
                s = ns
                reward += r

                # sample minibatch
                driver.buff.add(drv_s, drv_a[0], r, drv_ns, done)
                batch = driver.buff.getBatch(cfg.BATCH_SIZE)
                states, actions, rewards, new_states, dones = zip(*batch)

                # set target
                target_q = driver.critic.target_predict(new_states, driver.actor.target_predict(new_states))
                y = [rewards[i] + (cfg.GAMMA * target_q[i] if not dones[i] else 0) for i in range(len(batch))]

                # update critic
                driver.critic.train(y, states, actions)

                # update actor
                grads = driver.critic.gradients(states, driver.actor.predict(states))
                driver.actor.train(states, grads)

                # update the target networks
                driver.actor.target_train()
                driver.critic.target_train()

                # render
                env.metadata['extra'] = -drv_a[0]
                env.render()

            if ep > 0 and ep % save_every_episodes == 0:
                driver.save()

            print("%3d  RW = %+7.0f     NR = %.3f" % (ep, reward, noise_rate))

    @property
    def scope(self):
        name = self.prefix + '_' if self.prefix is not None else ''
        name += self.__class__.__name__
        return name.replace('-', '_')
