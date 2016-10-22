from collections import deque
import actor_network
import critic_network
from ou_noise import OUNoise
import numpy as np
import config
import random


class AgentDDPG:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.replay_memory = deque()
        self.actor_network = actor_network.ActorNetwork(state_size, action_size)
        self.critic_network = critic_network.CriticNetwork(state_size, action_size)

        self.ou_noise = OUNoise(action_size)

        self.time_step = 0

    def set_state(self, obs):
        self.state = obs

    def get_action(self):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.get_action(self.state)
        return np.clip(action + self.ou_noise.noise(), self.env.action_space.low, self.env.action_space.high)

    def set_feedback(self, obs, action, reward, done):
        next_state = obs
        self.replay_memory.append((self.state, action, reward, next_state, done))

        self.state = next_state
        self.time_step += 1

        if len(self.replay_memory) > config.MEMORY_SIZE:
            self.replay_memory.popleft()

        # Store transitions to replay start size then start training
        if self.time_step > config.OBSERVATION_STEPS:
            self.train()

        if self.time_step % config.SAVE_EVERY_X_STEPS == 0:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)

        # reinit the random process when an episode ends
        if done:
            self.ou_noise.reset()

    def train(self):
        minibatch = random.sample(self.replay_memory, config.MINI_BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        action_batch = np.resize(action_batch, [config.MINI_BATCH_SIZE, 1])

        # Calculate y
        y_batch = []
        next_action_batch = self.actor_network.get_target_action_batch(next_state_batch)
        q_value_batch = self.critic_network.get_target_q_batch(next_state_batch, next_action_batch)

        for i in range(0, config.MINI_BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + config.FUTURE_REWARD_DISCOUNT * q_value_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [len(y_batch), 1])

        # Update critic by minimizing the loss
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch = self.actor_network.get_action_batch(state_batch)
        q_gradient_batch = self.critic_network.get_gradients(state_batch, action_batch)

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()
