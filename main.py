from __future__ import print_function

import tensorflow as tf
import gym
import config as cfg
from algs.PeterKovacs.ddqn import DDQN
from algs.RomanKoshelev.sa_wrapper import SAWrapper


def main():
    # sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # env
    env = gym.make('Tentacle-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # agent
    # agent = DDQN(sess, env.spec.id, obs_dim, act_dim, cfg.CHECKPOINT_FOLDER)
    # agent.train(env, cfg.EPISODES, cfg.STEPS, cfg.SAVE_EPISODES)
    # agent.run(env, cfg.EPISODES, cfg.STEPS)

    agent = SAWrapper(sess, env.spec.id, obs_dim, act_dim, cfg.CHECKPOINT_FOLDER)
    agent.train(env, cfg.EPISODES, cfg.STEPS, cfg.SAVE_EPISODES)
    # agent.run(env, cfg.EPISODES, cfg.STEPS)

if __name__ == '__main__':
    main()
