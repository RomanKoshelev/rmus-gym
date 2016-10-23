from __future__ import print_function

import numpy as np
import algs.PeterKovacs.config as cfg
from algs.PeterKovacs.ddqn import DDQN
from algs.PeterKovacs.ou_noise import OUNoise


class SAWrapper:
    def __init__(self, sess, env_id, obs_dim, act_dim, data_folder):
        self.sess = sess
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.drv_dim = 2
        self.data_folder = data_folder
        self.exploration = OUNoise(self.drv_dim)

    def run(self, env, episodes, steps):
        print(self.__class__.__name__)
        agent = DDQN(self.sess, self.env_id, self.obs_dim, self.act_dim, self.data_folder)
        for ep in range(episodes):
            s = env.reset()
            reward = 0
            for t in range(steps):
                env.render()
                a = agent.act(s)
                s, r, _, _ = env.step(a[0])
                reward += r

            print("%3d  Reward = %+7.0f  " % (ep, reward))

    def train(self, env, episodes, steps, save_every_episodes):
        agent = DDQN(self.sess, self.env_id, self.obs_dim, self.act_dim, self.data_folder)
        # driver = DDQN(self.sess, self.env_id, self.drv_dim, self.drv_dim)

        for ep in range(episodes):
            s, reward, done = env.reset(), 0, False
            self.exploration.reset()

            for t in range(steps):
                env.render()

                # extract world state
                wn = self.drv_dim
                wrd_s = s[0:wn]
                inn_s = s[wn:]

                # execute step
                drv_a = [wrd_s[0], wrd_s[1]]  # driver.actor.predict([s]) + self.exploration.noise()
                agn_s = np.append(drv_a, inn_s)
                agn_a = agent.actor.predict([agn_s])
                ns, r, done, info = env.step(agn_a[0])
                wrd_ns = ns[0:wn]
                s = ns
                reward += r
                continue

                driver.buff.add(wrd_s, drv_a[0], r, wrd_ns, done)

                # sample minibatch
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

            if ep % save_every_episodes == 0:
                pass
                # driver.save()

            print("%3d  Reward = %+7.0f  " % (ep, reward))