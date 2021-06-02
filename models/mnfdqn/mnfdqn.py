### LIBRARIES ###
# Global libraries
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom libraries
from utils.utils import initialize_weights
from models.mnfdqn.mnflinear import MNFLinear


### CLASS DEFINITION ###
class MNFDQN(nn.Module):
    def __init__(
        self, input_dim, n_actions, hidden_dim, n_hidden, n_flows_q, n_flows_r
    ):
        super(MNFDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = MNFLinear(3136, 512, hidden_dim, n_hidden, n_flows_q, n_flows_r)
        self.fc2 = MNFLinear(512, n_actions, hidden_dim, n_hidden, n_flows_q, n_flows_r)
        initialize_weights(self)

    def forward(self, x, kl=True):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        if kl:
            x, kldiv1 = self.fc1(x, kl=True)
            x = F.relu(x)
            x, kldiv2 = self.fc2(x, kl=True)
            kldiv = kldiv1 + kldiv2
            return x, kldiv
        else:
            x = F.relu(self.fc1(x, kl=False))
            x = self.fc2(x, kl=False)
            return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
