
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, action_std = 0.0):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_size, 80),
            nn.Tanh(),
            nn.Linear(80, 40),
            nn.Tanh(),
            nn.Linear(40, action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 80),
            nn.Tanh(),
            nn.Linear(80, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        self.action_variance = torch.full((action_size,), action_std*action_std)
    def forward(self, x):
        raise NotImplementedError
    