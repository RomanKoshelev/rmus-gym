from __future__ import print_function
import numpy as np
import tensorflow as tf
import gym

import config as cfg
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise


def train(sess, env):
    action_dim = env.action_space.shape[0]
    input_dim = env.observation_space.shape[0]

    actor = ActorNetwork(sess, input_dim, action_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
    critic = CriticNetwork(sess, input_dim, action_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)
    buff = ReplayBuffer(cfg.BUFFER_SIZE)
    exploration = OUNoise(action_dim)

    for ep in range(cfg.EPISODES):
        s, _, done = env.reset(), 0, False
        reward = 0
        exploration.reset()

        for t in range(cfg.STEPS):
            env.render()

            a = actor.predict([s]) + exploration.noise()
            sn, r, done, info = env.step(a[0])
            buff.add(s, a[0], r, sn, done)

            # sample a random minibatch
            batch = buff.getBatch(cfg.BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])

            # set target
            target_q_values = critic.target_predict(new_states, actor.target_predict(new_states))
            y = [rewards[i] + (cfg.GAMMA * target_q_values[i] if not dones[i] else 0) for i in range(len(batch))]

            #            y = []
            # for i in range(len(batch)):
#                 if dones[i]:
#                     y.append(rewards[i])
#                 else:
#                     y.append(rewards[i] + cfg.GAMMA * target_q_values[i])

            # update critic
            critic.train(y, states, actions)

            # update actor
            grads = critic.gradients(states, actor.predict(states))
            actor.train(states, grads)

            # update the target networks
            actor.target_train()
            critic.target_train()

            # move to next state
            s = sn
            reward += r

        print("%3d  Reward = %+5.0f  " % (ep, reward))


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    env = gym.make(cfg.ENVIRONMENT_NAME)

    train(sess, env)


if __name__ == '__main__':
    main()
