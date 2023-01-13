import torch.nn as nn

class DQNnet(nn.Module):

    def __init__(self, state_length, n_actions):
        super(DQNnet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_length, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    def forward(self, x):
        return self.model(x)