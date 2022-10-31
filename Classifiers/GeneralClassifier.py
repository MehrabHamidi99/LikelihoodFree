import torch
import torch.nn as nn
from pyro.distributions import Normal, Gamma, MultivariateNormal


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
