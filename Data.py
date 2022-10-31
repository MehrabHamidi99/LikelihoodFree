import pickle
import numpy as np
import torch
import hamiltorch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

n_theta = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = open('data/AUTsummary.hg19_AE_30SNP.pkl', 'rb')
dataset = pickle.load(file)

data = dataset[4][0]
labels = dataset[4][1]

data = data[:, 0:100]

print(data.shape)
print(labels.shape)

log_norm_constant = -0.5 * np.log(2 * np.pi)


def log_gaussian(x, mean=0, logvar=0.):
    if type(logvar) == 'float':
        logvar = x.new(1).fill_(logvar)

    a = (x - mean) ** 2
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p = log_p + log_norm_constant

    return log_p


def get_likelihoods(X, mu, logvar, log=True):
    log_likelihoods = log_gaussian(
        X[None, :, :],
        mu[:, None, :],
        logvar[:, None, :]
    )

    log_likelihoods = log_likelihoods.sum(-1)

    if not log:
        log_likelihoods.exp_()

    return log_likelihoods


mean = torch.tensor(data.mean(axis=0))  # p*1
stddev = torch.tensor(np.cov(data.T))  # p*p
dist = MultivariateNormal(loc=mean, covariance_matrix=stddev)
new_batch = dist.rsample()
