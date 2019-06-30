import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DQN,self).__init__()
        torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.activation = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.activation(x)

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DuelingDQN,self).__init__()
        torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value_fc1 = nn.Linear(128, 32)
        self.value_activation = nn.Linear(32, 1)
        self.advantage_fc1 = nn.Linear(128, 32)
        self.advantage_activation = nn.Linear(32, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = F.relu(self.value_fc1(x))
        v = self.value_activation(v).expand(x.size(0), self.action_size)
        a = F.relu(self.advantage_fc1(x))
        a = self.advantage_activation(a)
        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x
