from __future__ import print_function

import os
import tensorflow as tf
import gym

BASE_PATH = '../out/tests/'


def launch(proc, agent_class, env_name, episodes, steps, save_every_episodes, reuse_weights=False):
    def func_name():
        import traceback
        return traceback.extract_stack(None, 3)[0][2]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    path = os.path.join(BASE_PATH, func_name())
    if not reuse_weights and os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

    agent = agent_class(sess, env.spec.id, obs_dim, act_dim, path)
    if proc == 'train':
        agent.train(env, episodes, steps, save_every_episodes)
    elif proc == 'run':
        agent.run(env, episodes, steps)


def test_train_tentacle():
    from alg.PeterKovacs.ddgp import DDGP
    launch('train', DDGP, 'Tentacle-v0', episodes=1000, steps=100, save_every_episodes=10)


if __name__ == '__main__':
    test_train_tentacle()
