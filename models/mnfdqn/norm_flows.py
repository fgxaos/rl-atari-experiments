### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
from torch.autograd import Variable

# Custom libraries
from utils.utils import initialize_weights

### CLASS DEFINITIONS ###
class SingleMaskedNVPFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden):
        super(SingleMaskedNVPFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        hidden_modules = []
        for _ in range(n_hidden):
            hidden_modules.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_modules.append(nn.Tanh())

        self.hidden_layer = nn.Sequential(*hidden_modules)
        self.mu_layer = nn.Linear(hidden_dim, input_dim)
        self.sigma_layer = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
        self.register_buffer("mask", torch.Tensor(input_dim))
        initialize_weights(self)

    def reset_noise(self):
        mask = torch.bernoulli(0.5 * torch.ones(self.input_dim))
        self.mask.copy_(mask)

    def forward(self, z, kl=True):
        if self.training:
            mask = self.mask
        else:
            mask = 0.5

        h = self.first_layer(mask * z)
        h = self.hidden_layer(h)
        mu = self.mu_layer(h)
        sigma = self.sigma_layer(h)
        z = (1 - mask) * (z * sigma + (1 - sigma) * mu) + mask * z
        if kl:
            if z.dim() == 1:
                logdet = ((1 - mask) * torch.log(sigma)).sum()
            else:
                logdet = ((1 - mask) * torch.log(sigma)).sum(1)
            return z, logdet
        else:
            return z


class MaskedNVPFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden, n_flows):
        super(MaskedNVPFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.flow_list = nn.ModuleList(
            [
                SingleMaskedNVPFlow(input_dim, hidden_dim, n_hidden)
                for _ in range(n_flows)
            ]
        )

    def forward(self, z, kl=True):
        if kl:
            if z.dim() == 1:
                logdets = 0
            else:
                logdets = Variable(torch.zeros_like(z[:, 0]))
            for flow in self.flow_list:
                z, logdet = flow(z, kl=True)
                logdets += logdet
            return z, logdets
        else:
            for flow in self.flow_list:
                z = flow(z, kl=False)
            return z

    def reset_noise(self):
        for flow in self.flow_list:
            flow.reset_noise()
