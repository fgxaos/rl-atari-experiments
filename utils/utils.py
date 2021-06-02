### LIBRARIES ###
# Global libraries
import math

import torch
import torch.nn as nn

### FUNCTION DEFINITIONS ###


def initialize_weights(model):
    """Initializes the weights of the given model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels + m.in_channels
            m.weight.data.normal_(0, math.sqrt(4.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.in_features + m.out_features
            m.weight.data.normal_(0, math.sqrt(4.0 / n))
            m.bias.data.zero_()
