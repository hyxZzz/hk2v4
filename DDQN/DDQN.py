import torch
import torch.nn as nn


class Double_DQN(nn.Module):
    def __init__(self, state_size: int, action_size_each: int, num_agents: int = 2):
        super(Double_DQN, self).__init__()

        hidden1 = 512
        hidden2 = 1024

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(hidden2, action_size_each) for _ in range(num_agents)])

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.shared(state)
        q_values = tuple(head(features) for head in self.heads)
        return q_values
