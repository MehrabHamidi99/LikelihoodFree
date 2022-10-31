import numpy as np
import torch
from pyro.distributions import Normal, Gamma, MultivariateNormal


class Metropolis_Hastings(object):

    def __int__(self, alpha_gam, beta_gam, var, rep):

        self.alpha_gam = alpha_gam
        self.beta_gam = beta_gam
        self.var = var
        self.REP = rep

    def prior_sample(self, size=1):
        gamma_dist = Gamma(self.alpha_gam, 1 / self.beta_gam)
        sigma = gamma_dist.sample(sample_shape=torch.Size([size]))

        normal_dist = Normal(0, sigma)
        beta = normal_dist.sample()
        return beta, sigma

    def prior_prob(self, beta, sigma):
        gamma_dist = Gamma(self.alpha_gam, 1 / self.beta_gam, validate_args=True)
        p_sig = torch.exp(gamma_dist.log_prob(sigma))
        normal_dist = Normal(0, sigma)
        p_beta = torch.exp(normal_dist.log_prob(beta))

        return p_beta, p_sig

    # from scipy.stats import multivariate_normal

    def metropolis_heisting(self, cur_beta, cur_sigma, cur_prob_theta, geno_mat, h_classifier):
        #     changed_value = np.full(cur_theta_prob.shape, 1)
        counter = 0

        while True:
            new_beta = MultivariateNormal(cur_beta.float(), self.var.float()).sample()
            new_sigma = MultivariateNormal(cur_sigma.float(), self.var.float()).sample()

            # new_beta = torch.clip(new_beta, -2, 2)

            beta_prob, sigma_prob = self.prior_prob(new_beta, cur_sigma)
            new_ys = torch.matmul(geno_mat.float(), new_beta)
            with torch.no_grad():
                r_value = h_classifier(new_ys, geno_mat, new_beta, cur_sigma)
                r_value = torch.mean(r_value, dim=0)
                r_value = r_value / (1 - r_value)
                new_theta_prob = r_value * beta_prob * sigma_prob

                rho = ((new_theta_prob) / (cur_prob_theta + 1e-7)).cpu().detach().numpy()
                rho[rho > 1] = 1
                #         rho = rho * changed_value
                #        rho = min(1, (new_theta_prob)/(cur_theta_prob))
                random_num = np.random.rand()
                if np.isnan(np.sum(rho)):
                    print("EROR ------------------------")
                    print(np.isnan(np.sum(cur_prob_theta)))
                    print(np.isnan(np.sum(new_theta_prob)))

                if random_num < np.mean(rho):
                    cur_beta = new_beta
                    cur_sigma = cur_sigma
                    cur_theta_prob = new_theta_prob
                    # cur_beta = 1 / (1 + np.exp(-1 * cur_beta))
                    return cur_sigma, cur_beta, cur_theta_prob, new_ys

            counter += 1

            if counter > self.REP:
                #             print("It Takes quite time :(")
                cur_beta = (new_beta + cur_beta) / 2
                cur_sigma = (new_sigma + cur_sigma) / 2
                cur_theta_prob = (new_theta_prob + cur_prob_theta) / 2
                # cur_beta = 1 / (1 + np.exp(-1 * cur_beta))
                return cur_sigma, cur_beta, cur_theta_prob, new_ys
