from __future__ import print_function

from algs.PeterKovacs.ddqn import DDQN


class SaWrapper:
    def __init__(self, sess, env_id, obs_dim, act_dim, data_folder):
        self.sess = sess
        self.env_id = env_id
        self.agent = DDQN(sess, env_id, obs_dim, act_dim, data_folder)

    def run(self, env, episodes, steps):
        print(self.__class__.__name__)
        for ep in range(episodes):
            s = env.reset()
            reward = 0
            for t in range(steps):
                env.render()
                a = self.agent.act(s)
                s, r, _, _ = env.step(a[0])
                reward += r

            print("%3d  Reward = %+7.0f  " % (ep, reward))

    def train(self, episodes, steps):
        pass
