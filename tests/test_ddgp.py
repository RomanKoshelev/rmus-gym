from __future__ import print_function

import os
import tensorflow as tf
import gym
from alg.PeterKovacs.ddpg import DDPG
import numpy as np

BASE_PATH = '../out/tests/'
RANDOM_SEED = 2016


def launch(proc, agent_class, env_name, episodes=125000, steps=None, save_every_episodes=100, reuse_weights=False):
    def func_name():
        import traceback
        return traceback.extract_stack(None, 3)[0][2]

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

    agent = agent_class(sess, env.spec.id, obs_dim, obs_box, act_dim, act_box, path)
    if proc == 'train':
        agent.train(env, episodes, steps, save_every_episodes)
    elif proc == 'run':
        agent.run(env, episodes, steps)


def Tentacle(from_scratch=False):
    launch('train', DDPG, 'Tentacle-v0', reuse_weights=not from_scratch)


def Ant(from_scratch=False):
    launch('train', DDPG, 'Ant-v1', reuse_weights=not from_scratch)


def Reacher(from_scratch=False):
    launch('train', DDPG, 'Reacher-v1', reuse_weights=not from_scratch)


def HumanoidStandup(from_scratch=False):
    launch('train', DDPG, 'HumanoidStandup-v1', reuse_weights=not from_scratch)


def Pendulum(from_scratch=False):
    # OK
    # launch('train', DDPG, 'Pendulum-v0', steps=100, save_every_episodes=10, reuse_weights=not from_scratch)
    launch('run', DDPG, 'Pendulum-v0', steps=100, reuse_weights=True)


def InvertedDoublePendulum(from_scratch=False):
    # OK
    # launch('train', DDPG, 'InvertedDoublePendulum-v1', reuse_weights=not from_scratch)
    launch('run', DDPG, 'InvertedDoublePendulum-v1', reuse_weights=True)

# LunarLanderContinuous-v2

if __name__ == '__main__':
    # Pendulum(from_scratch=True)
    # InvertedDoublePendulum(from_scratch=True)
    Tentacle(from_scratch=True)
    # Reacher(from_scratch=True)
    # ant()
    # humanoid_standup()
