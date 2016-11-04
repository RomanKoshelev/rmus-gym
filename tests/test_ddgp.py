from __future__ import print_function

import os
import tensorflow as tf
import gym
from alg.PeterKovacs.ddpg import DDPG
import numpy as np

BASE_PATH = '../out/tests/'
RANDOM_SEED = 2016


def launch(proc, env_name, episodes=125000, steps=None, save_every_episodes=100, reuse_weights=False):
    def func_name():
        import traceback
        return traceback.extract_stack(None, 4)[0][2]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    env = gym.make(env_name)

    if steps is None:
        steps = env.spec.timestep_limit

    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

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

    agent = DDPG(sess, env.spec.id, obs_dim, obs_box, act_dim, act_box, path)
    if proc == 'train':
        agent.train(env, episodes, steps, save_every_episodes)
    elif proc == 'run':
        agent.run(env, episodes, steps)

    env.close()


def run(env_name, episodes=1000, steps=None):
    launch("run", env_name, episodes, steps, save_every_episodes=0, reuse_weights=True)


def train(env_name, episodes=125000, steps=None, save_every_episodes=100, reuse_weights=False):
    launch("train", env_name, episodes, steps, save_every_episodes, reuse_weights)

# =================================================================================================================


# PASSED
def Pendulum():
    env = 'Pendulum-v0'
    train(env, steps=100, save_every_episodes=50)
    # run(env, steps=100)


# PASSED
def InvertedDoublePendulum():
    env = 'InvertedDoublePendulum-v1'
    train(env)
    # run(env)


def Tentacle():
    env = 'Tentacle-v0'
    train(env)
    # run(env)


def Reacher():
    env = 'Reacher-v1'
    train(env)
    # run(env)


def Ant():
    env = 'Ant-v1'
    train(env)
    # run(env)


def HumanoidStandup():
    env = 'HumanoidStandup-v1'
    train(env)
    # run(env)


# LunarLanderContinuous-v2

if __name__ == '__main__':
    # Pendulum()
    # InvertedDoublePendulum()
    # Tentacle()
    Reacher()
    # Ant()
    # HumanoidStandup()
