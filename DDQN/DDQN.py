import torch
import torch.nn as nn



class Double_DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(Double_DQN, self).__init__()

        hidden_1 = 256
        hidden_2 = 256
        hidden_3 = 128

        self.fc1 = nn.Linear(state_size, hidden_1)
        self.norm1 = nn.LayerNorm(hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.norm2 = nn.LayerNorm(hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.out = nn.Linear(hidden_3, action_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in (self.fc1, self.fc2, self.fc3, self.out):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        out = self.activation(self.norm1(self.fc1(state)))
        out = self.activation(self.norm2(self.fc2(out)))
        out = self.dropout(self.activation(self.fc3(out)))
        q = self.out(out)
        return q
