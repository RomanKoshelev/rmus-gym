from __future__ import print_function

import os
import tensorflow as tf
import gym
from alg.PeterKovacs.ddpg import DDPG

BASE_PATH = '../out/tests/'


def launch(proc, agent_class, env_name, episodes=125000, steps=3000, save_every_episodes=100, reuse_weights=False):
    def func_name():
        import traceback
        return traceback.extract_stack(None, 3)[0][2]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_box = [env.observation_space.low, env.observation_space.high]
    act_box = [env.action_space.low, env.action_space.high]

    path = os.path.join(BASE_PATH, func_name())
    if not reuse_weights and os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

    agent = agent_class(sess, env.spec.id, obs_dim, obs_box, act_dim, act_box, path)
    if proc == 'train':
        agent.train(env, episodes, steps, save_every_episodes)
    elif proc == 'run':
        agent.run(env, episodes, steps)


def Tentacle():
    launch('train', DDPG, 'Tentacle-v0', episodes=20000, steps=100, reuse_weights=True)


def Ant():
    launch('train', DDPG, 'Ant-v1', episodes=10000, steps=200, reuse_weights=True)


def Reacher():
    launch('train', DDPG, 'Reacher-v1', episodes=10000, steps=100, reuse_weights=True)


def HumanoidStandup():
    launch('train', DDPG, 'HumanoidStandup-v1', episodes=10000, steps=100, reuse_weights=True)


def Pendulum():
    # OK
    launch('train', DDPG, 'Pendulum-v0', episodes=10000, steps=200, reuse_weights=True)


def InvertedDoublePendulum():
    launch('train', DDPG, 'InvertedDoublePendulum-v1', episodes=10000)


# LunarLanderContinuous-v2

if __name__ == '__main__':
    InvertedDoublePendulum()
    # tentacle()
    # ant()
    # reacher()
    # humanoid_standup()
    # pendulum()
