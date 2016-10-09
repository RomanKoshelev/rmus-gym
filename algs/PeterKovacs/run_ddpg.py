import numpy as np
import tensorflow as tf
import gym
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from ou_noise import OUNoise
import timeit

# https://gym.openai.com/evaluations/eval_xjuUFdvrQR68YWvUqsjKPQ#writeup
# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING http://arxiv.org/pdf/1509.02971v5.pdf
# https://www.youtube.com/watch?v=tJBIqkC1wWM&feature=youtu.be

# REPLAY BUFFER CONSTS
BUFFER_SIZE = 10000  # 10000
BATCH_SIZE = 128  # 128
# FUTURE REWARD DECAY
GAMMA = 0.99  # 0.99
# TARGET NETWORK UPDATE STEP
TAU = 0.001  # 0.001
# LEARNING_RATE
LRA = 0.0001  # 0.0001
LRC = 0.001  # 0.001
# ENVIRONMENT_NAME
ENVIRONMENT_NAME = 'Tentacle-v0'
# L2 REGULARISATION
L2C = 0.01
L2A = 0

env = gym.make(ENVIRONMENT_NAME)
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

input_dim = env.observation_space.shape[0]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess = tf.InteractiveSession()

actor = ActorNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRA, L2A)
critic = CriticNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRC, L2C)
buff = ReplayBuffer(BUFFER_SIZE)
# exploration = OUNoise(action_dim)

env.monitor.start('experiments/' + ENVIRONMENT_NAME, force=True)

for ep in range(100000):
    # open up a game state
    s_t, r_0, done = env.reset(), 0, False

    reward = 0

    exploration_noise = OUNoise(action_dim)

    # exploration.reset()
    for t in range(100):
        env.render()

        # select action according to current policy and exploration noise
        a_t = actor.predict([s_t]) + exploration_noise.noise()

        # execute action and observe reward and new state
        s_t1, r_t, done, info = env.step(a_t[0])

        # store transition in replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, done)

        # sample a random minibatch of N transitions (si, ai, ri, si+1) from replay buffer
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        # set target yi = ri + gamma*target_critic_network(si+1, target_actor_network(si+1))
        target_q_values = critic.target_predict(new_states, actor.target_predict(new_states))

        y_t = []
        for i in range(len(batch)):
            if dones[i]:
                y_t.append(rewards[i])
            else:
                y_t.append(rewards[i] + GAMMA * target_q_values[i])

        # update critic network by minimizing los L = 1/N sum(yi - critic_network(si,ai))**2
        critic.train(y_t, states, actions)

        # update actor policy using sampled policy gradient
        a_for_grad = actor.predict(states)
        grads = critic.gradients(states, a_for_grad)
        actor.train(states, grads)

        # update the target networks
        actor.target_train()
        critic.target_train()

        # move to next state
        s_t = s_t1
        reward += r_t

    print ("%3d  Reward = %+10.0f  " % (ep, reward))

# Dump result info to disk
env.monitor.close()
