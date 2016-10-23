from __future__ import print_function

import tensorflow as tf
import gym
import config as cfg
from algs.PeterKovacs.ddqn import DDQN


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    env = gym.make(cfg.ENVIRONMENT_NAME)
    agent = DDQN(sess, env, cfg.CHECKPOINT_FOLDER)

    agent.train()
    agent.run()


if __name__ == '__main__':
    main()
