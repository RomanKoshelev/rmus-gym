from __future__ import print_function

import os

import tensorflow as tf
import gym

import config as cfg
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise


def train(sess, env):
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    actor = ActorNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
    critic = CriticNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)
    buff = ReplayBuffer(cfg.BUFFER_SIZE)
    exploration = OUNoise(act_dim)

    load(sess)

    for ep in range(cfg.EPISODES):
        s, reward, done = env.reset(), 0, False
        exploration.reset()

        for t in range(cfg.STEPS):
            env.render()

            # execute step
            a = actor.predict([s]) + exploration.noise()
            ns, r, done, info = env.step(a[0])
            buff.add(s, a[0], r, ns, done)

            # sample minibatch
            batch = buff.getBatch(cfg.BATCH_SIZE)
            states, actions, rewards, new_states, dones = zip(*batch)

            # set target
            target_q = critic.target_predict(new_states, actor.target_predict(new_states))
            y = [rewards[i] + (cfg.GAMMA * target_q[i] if not dones[i] else 0) for i in range(len(batch))]

            # update critic
            critic.train(y, states, actions)

            # update actor
            grads = critic.gradients(states, actor.predict(states))
            actor.train(states, grads)

            # update the target networks
            actor.target_train()
            critic.target_train()

            # move to next state
            s = ns
            reward += r

        if ep % cfg.SAVE_EVERY_EPISODES == 0:
            save(sess)

        print("%3d  Reward = %+7.0f  " % (ep, reward))


def run(sess, env):
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    actor = ActorNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRA, cfg.L2A)
    critic = CriticNetwork(sess, obs_dim, act_dim, cfg.BATCH_SIZE, cfg.TAU, cfg.LRC, cfg.L2C)

    load(sess)

    for ep in range(cfg.EPISODES):
        s = env.reset()
        reward = 0

        for t in range(cfg.STEPS):
            env.render()

            # execute step
            a = actor.predict([s])
            s, r, done, info = env.step(a[0])

            # move to next state
            reward += r

        print("%3d  Reward = %+7.0f  " % (ep, reward))


def save(sess):
    print("Saving...")
    tf.train.Saver().save(sess, cfg.CHECKPOINT_PATH)


def load(sess):
    if os.path.exists(cfg.CHECKPOINT_PATH):
        tf.train.Saver().restore(sess, cfg.CHECKPOINT_PATH)
        print ("Successfully loaded:", cfg.CHECKPOINT_PATH)
    else:
        print ("Could not find old network weights for ", cfg.CHECKPOINT_PATH)


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    env = gym.make(cfg.ENVIRONMENT_NAME)

    # train(sess, env)
    run(sess, env)


if __name__ == '__main__':
    main()
