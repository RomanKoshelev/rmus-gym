from __future__ import print_function

import os

import tensorflow as tf
import gym


def launch(proc, agent_type, agent_class, env_name, episodes=25000, steps=100):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = agent_class(sess, env.spec.id, obs_dim, act_dim, os.path.join('./dat/', agent_type))
    if proc == 'train':
        agent.train(env, episodes, steps, save_every_episodes=100)
    elif proc == 'run':
        agent.run(env, episodes, steps)

