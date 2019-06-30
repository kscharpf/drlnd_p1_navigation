import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from model import DQN, DuelingDQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rl_config import RLConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self,
            state_size,
            action_size,
            config=RLConfig()):
        self.seed = random.seed(config.seed)
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = config.batch_size
        self.batch_indices = torch.arange(config.batch_size).long().to(device)
        self.samples_before_learning = config.samples_before_learning
        self.learn_interval = config.learning_interval
        self.parameter_update_interval = config.parameter_update_interval
        self.per_epsilon = config.per_epsilon
        self.tau = config.tau
        self.gamma = config.gamma

        if config.useDuelingDQN:
            self.qnetwork_local = DuelingDQN(state_size, action_size, config.seed).to(device)
            self.qnetwork_target = DuelingDQN(state_size, action_size, config.seed).to(device)
        else:
            self.qnetwork_local = DQN(state_size, action_size, config.seed).to(device)
            self.qnetwork_target = DQN(state_size, action_size, config.seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.learning_rate)

        self.doubleDQN = config.useDoubleDQN
        self.usePER = config.usePER
        if self.usePER:
            self.memory = PrioritizedReplayBuffer(config.buffer_size,config.per_alpha)
        else:
            self.memory = ReplayBuffer(config.buffer_size)

        self.t_step = 0

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done, beta):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % self.learn_interval == 0:
            if len(self.memory) > self.samples_before_learning:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
                target = self.qnetwork_local(state).data
                old_val = target[0][action]
                target_val = self.qnetwork_target(next_state).data
                if done:
                    target[0][action] = reward
                else:
                    target[0][action] = reward + self.gamma * torch.max(target_val)
                if self.usePER:
                    states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, beta)
                else:
                    indices=None
                    weights=None
                    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

                self.learn(states, actions, rewards, next_states, dones, indices, weights, self.gamma)

    def learn(self, states, actions, rewards, next_states, dones, indices, weights, gamma):
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones.astype(np.uint8))).float().to(device)
        Q_targets_next = self.qnetwork_target(next_states).detach()

        if self.doubleDQN:
            # choose the best action from the local network
            next_actions = self.qnetwork_local(next_states).argmax(dim=-1)
            Q_targets_next = Q_targets_next[self.batch_indices, next_actions]
        else:
            Q_targets_next = Q_targets_next.max(1)[0]

        Q_targets = rewards + gamma * Q_targets_next.reshape((self.batch_size, 1)) * (1 - dones)

        pred = self.qnetwork_local(states)
        Q_expected = pred.gather(1, actions)

        if self.usePER:
            errors = torch.abs(Q_expected - Q_targets).data.numpy() + self.per_epsilon
            self.memory.update_priorities(indices, errors)

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_expected, Q_targets)
        loss.backward()
        self.optimizer.step()

        if self.t_step % self.parameter_update_interval == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, qnetwork_local, qnetwork_target, tau):
        for local_param, target_param in zip(qnetwork_local.parameters(), qnetwork_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



