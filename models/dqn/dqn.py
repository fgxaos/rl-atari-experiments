### LIBRARIES ###
# Global libraries
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom libraries
from utils.utils import initialize_weights

### CLASS DEFINITION ###
class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_actions)
        initialize_weights(self)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
