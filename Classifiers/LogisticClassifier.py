import GeneralClassifier
from GeneralClassifier import *


class LogisticRegression(nn.Module):

    def __init__(self, alpha_gam, beta_gam, num_classes=1):
        super(LogisticRegression, self).__init__()

        self.alpha_gam = alpha_gam
        self.beta_gam = beta_gam

        self.discriminator = nn.Sequential(nn.Conv1d(1, 32, kernel_size=6, stride=2),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Dropout(0.5),
                                           nn.Conv1d(32, 16, kernel_size=6, stride=3),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Dropout(0.3),
                                           nn.Conv1d(16, 4, kernel_size=6, stride=4),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.Dropout(0.1),
                                           nn.Conv1d(4, 1, kernel_size=5, dilation=1),
                                           nn.LeakyReLU(0.2, inplace=True))

        self.layer_out = nn.Sequential(nn.Linear(6, 1),
                                       nn.Sigmoid())

        self.criterion = torch.nn.BCELoss(size_average=False)

    def forward(self, y, genotypes_batch, beta, sigma, label=None, testing=False):

        y = torch.tensor(y).to(self.device).float()
        beta = torch.tensor(beta).to(self.device).float()
        sigma = torch.tensor(sigma).to(self.device).float()

        gamma_dist = Gamma(self.alpha_gam, 1 / self.beta_gam)
        sigma_n = gamma_dist.sample(sample_shape=sigma.shape)

        normal_dist = Normal(0, sigma_n)
        beta_f = normal_dist.sample()
        genotypes_batch = torch.tensor(genotypes_batch).to(self.device).float()

        pheno_fake_f = torch.matmul(genotypes_batch, beta)[:, None]
        pheno_fake_s = torch.matmul(genotypes_batch, beta_f)[:, None]
        y = y[:, None]

        # y = torch.Tensor(y).to(device)
        # beta = torch.Tensor(beta).to(device)

        # y = torch.repeat_interleave(y[:, None], d, dim=-1)
        # beta = torch.repeat_interleave(beta[None, :], y.shape[0], dim=0)

        x = torch.cat([beta, sigma], dim=-1)[None, None, :]

        #         out = self.linear(x)
        #         out = self.activation(out)
        x = self.discriminator(x).squeeze()

        x = torch.repeat_interleave(x[None, :], y.shape[0], dim=0)
        lab_vec = torch.cat([y, pheno_fake_f, pheno_fake_s], dim=-1)
        aux_loss = nn.MSELoss()(x, lab_vec)
        x = torch.cat([x, lab_vec], dim=-1)

        # x = self.batchnorm1(x)

        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        # x = torch.squeeze(self.layer_out(x))

        probs = self.layer_out(x)

        # print(x[x>0].shape, "CLASSIFIER")
        # input()

        if label is not None:

            loss = self.criterion(probs, label[:, None])

            if testing:
                predicted = torch.round(probs).squeeze()
                print(float((len(label) - torch.sum(torch.abs(predicted - label)))) / len(label) * 100.0, " % acc")

                return torch.sum(torch.abs(predicted - label))

            return probs, loss + aux_loss

        return probs
