import torch
import torch.nn as nn
import pyro.nn
from pyro.distributions import Normal, Binomial


class SNP_prediction_model(pyro.nn.PyroModule):
    def __init__(self, input_dim):
        super(SNP_prediction_model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        pyro.nn.module.to_pyro_module_(self.model)

        # Fully bayesian!
        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, pyro.nn.PyroSample(prior=pyro.distributions.Normal(0, 1)
                                                    .expand(value.shape)
                                                    .to_event(value.dim())))

    def forward(self, z, w):

        snps, dim = w.shape
        N, _ = z.shape

        w_input = torch.repeat_interleave(w.reshape([1, snps, dim]), N, dim=0)
        z_input = torch.repeat_interleave(torch.reshape(z, [N, 1, dim]), snps, dim=1)

        print(w_input.shape)
        print(z_input.shape)

        out = torch.cat([w_input, z_input], dim=2)

        out = self.model(out)

        return out.reshape([N, snps])


class trait_prediction_model(pyro.nn.PyroModule):
    def __init__(self, input_dim, dim):
        super(trait_prediction_model, self).__init__()

        self.model_f = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
        )
        self.return_model = nn.Linear(256 + dim, 1)

        pyro.nn.module.to_pyro_module_(self.model_f)
        pyro.nn.module.to_pyro_module_(self.return_model)

        # Fully bayesian!
        for m in self.model_f.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, pyro.nn.PyroSample(prior=pyro.distributions.Normal(0, 1)
                                                    .expand(value.shape)
                                                    .to_event(value.dim())))

        for m in self.return_model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, pyro.nn.PyroSample(prior=pyro.distributions.Normal(0, 1)
                                                    .expand(value.shape)
                                                    .to_event(value.dim())))

    def forward(self, z, x):

        N, _ = z.shape

        epsilon = Normal(loc=0.0, scale=1.0).sample(sample_shape=[N, 1])

        out = torch.cat([z, x, epsilon], dim=1)

        out = self.model_f(out)
        out = torch.cat([z, out], dim=1)

        out = self.return_model(out)

        out.Trace()

        return torch.reshape(out, [N])


if __name__ == '__main__':
    N = 5
    M = 100
    K = 2

    snp_net = SNP_prediction_model(2 * K)
    trait_net = trait_prediction_model(M + K + 1, K)

    z = Normal(loc=0.0, scale=1.0).sample([N, K])
    w = Normal(loc=0.0, scale=1.0).sample([M, K])

    logits = snp_net(z, w)

    x = Binomial(total_count=2.0, logits=logits).sample()

    y = trait_net(z, x)

    print(y)
    print(y.shape)
import numpy as np
np.random.negative_binomial()


