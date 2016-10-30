from __future__ import print_function

import os

import tensorflow as tf
import gym


def launch(proc, data_folder, agent_class, env_name, episodes=1025000, steps=100):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = agent_class(sess, env.spec.id, obs_dim, act_dim, os.path.join('./dat/', data_folder))
    if proc == 'train':
        agent.train(env, episodes, steps, save_every_episodes=100)
    elif proc == 'run':
        agent.run(env, episodes, steps)

if __name__ == '__main__':
    # from alg.RomanKoshelev.sa_wrapper import SAWrapper
    # launch('train', 'superagent/6', SAWrapper, 'Tentacle-v0', steps=100)

    from alg.PeterKovacs.ddgp import DDGP
    launch('train', 'unit', DDGP, 'Walker2d-v1', steps=300)
    # launch('run', 'unit', DDQN, 'Reacher-v1', steps=100)
