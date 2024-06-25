import torch
from torch import nn
from torch.nn.functional import softplus, softmax
import numpy as np
import yaml


class InvertiblePriorLinear(nn.Module):
    """docstring for InvertiblePrior"""

    def __init__(self):
        super(InvertiblePriorLinear, self).__init__()
        self.p = nn.Parameter(torch.rand([2]))

    def forward(self, eps):
        o = self.p[0] * eps + self.p[1]
        return o

    def inverse(self, o):
        eps = (o - self.p[1]) / self.p[0]
        print("inverse path eps")
        print(eps.size())
        return eps


class InvertiblePWL(nn.Module):
    """docstring for InvertiblePrior"""

    def __init__(self, vmin=-5, vmax=5, n=100, use_bias=True):
        super(InvertiblePWL, self).__init__()
        self.p = nn.Parameter(torch.randn([n + 1]) / 5)
        self.int_length = (vmax - vmin) / (n - 1)

        self.n = n
        if use_bias:
            self.b = nn.Parameter(torch.randn([1]) + vmin)
        else:
            self.b = vmin
        self.points = nn.Parameter(torch.from_numpy(np.linspace(vmin, vmax, n).astype('float32')).view(1, n),
                                   requires_grad=False)

    def to_positive(self, x):
        return torch.exp(x) + 1e-3

    def forward(self, eps):
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(eps.device)

        delta_bias[0] = self.b
        for i in range(self.n - 1):
            delta_bias[i + 1] = delta_bias[i] + delta_h[i]
        index = torch.sum(((eps - self.points) >= 0).long(), 1).detach()  # b * 1 from 0 to n

        start_points = index - 1
        start_points[start_points < 0] = 0
        delta_bias = delta_bias[start_points]
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach()
        delta_x = eps - start_points.view(-1, 1)

        k = self.to_positive(self.p[index])
        delta_fx = delta_x * k.view(-1, 1)

        o = delta_fx + delta_bias.view(-1, 1)

        return o

    def inverse(self, o):
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(o.device)
        delta_bias[0] = self.b
        for i in range(self.n - 1):
            delta_bias[i + 1] = delta_bias[i] + delta_h[i]
        index = torch.sum(((o - delta_bias) >= 0).long(), 1).detach()  # b * 1 from 0 to n
        start_points = index - 1
        start_points[start_points < 0] = 0
        delta_bias = delta_bias[start_points]
        intervel_incre = o - delta_bias.view(-1, 1)
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach()
        k = self.to_positive(self.p[index])
        delta_x = intervel_incre / k.view(-1, 1)
        eps = delta_x + start_points.view(-1, 1)
        return eps


class InvertiblePriorInv(nn.Module):
    """docstring for InvertiblePrior"""

    def __init__(self, prior):
        super(InvertiblePriorInv, self).__init__()
        self.prior = prior

    def forward(self, o):
        return self.prior.inverse(o)

    def inverse(self, eps):
        return self.prior(eps)

class SCM(nn.Module):
    def __init__(self, d, A=None, scm_type='mlp'):
        print("initializing the causal model...")
        super().__init__()
        self.d = d
        self.A_given = A
        self.A_fix_idx = A == 0

        config_file = open('config/model_config.yaml', 'r')
        config_service = yaml.safe_load(config_file)

        use_fixed_A = config_service['use_fixed_A']

        self.A = nn.Parameter(torch.zeros(d, d)) if not use_fixed_A else torch.tensor(A, dtype=torch.float32)

        # Elementwise nonlinear mappings
        if scm_type == 'linscm':
            prior_net_model = lambda: InvertiblePriorLinear()
            prior_net_enc_model = lambda x: InvertiblePriorInv(x)
        elif scm_type == 'nlrscm':
            prior_net_model = lambda: InvertiblePWL()
            prior_net_enc_model = lambda x: InvertiblePriorInv(x)
        else:
            raise NotImplementedError("Not supported prior network.")

        for i in range(d):
            setattr(self, "prior_net%d" % i, prior_net_model())
            setattr(self, "enc_net%d" % i, prior_net_enc_model(getattr(self, "prior_net%d" % i)))

    def set_zero_grad(self):
        if self.A_given is None:
            pass
        else:
            for i in range(self.d):
                for j in range(self.d):
                    if self.A_fix_idx[i, j]:
                        self.A.grad.data[i, j].zero_()

    def prior_nlr(self, z_mu, z_sigma):
        '''Nonlinear transformation f_2(z)'''
        zs_mu = torch.split(z_mu, 1, dim=1)
        zs_sigma = torch.split(z_sigma, 1, dim=1)

        z_mu_new = []
        z_sigma_new = []

        # Apply the nonlinear transformation to each part of z1
        for i in range(len(zs_mu)):
            z_mu_new.append(getattr(self, "prior_net%d" % i)(zs_mu[i]))

        # Apply the nonlinear transformation to each part of z2
        for i in range(len(zs_sigma)):
            z_sigma_new.append(getattr(self, "prior_net%d" % i)(zs_sigma[i]))

        # Concatenate the results along the second dimension
        causal_z_mu = torch.cat(z_mu_new, dim=1)
        causal_z_sigma = torch.cat(z_sigma_new, dim=1)

        return causal_z_mu, causal_z_sigma
    def enc_nlr(self, z):
        '''f_2^{-1}(z)'''
        zs = torch.split(z, 1, dim=1)
        z_new = []
        for i in range(self.d):
            z_new.append(getattr(self, "enc_net%d" % i)(zs[i]))
        return torch.cat(z_new, dim=1)

    def mask(self, z):  # Az
        z = torch.matmul(z, self.A)
        print(self.A)
        return z

    def inv_cal(self, mu_params, sigma_params):  # (I-A)^{-1}*eps
        adj_normalized = torch.inverse(torch.eye(self.A.shape[0], device=self.A.device) - self.A)
        z_mean = torch.matmul(mu_params, adj_normalized)
        z_sigma = torch.matmul(sigma_params, adj_normalized)
        return z_mean, z_sigma

    def get_eps(self, z):
        '''Returns epsilon from f_2^{-1}(z)'''
        return torch.matmul(z, torch.eye(self.A.shape[0], device=self.A.device) - self.A)

    def intervene(self, z, z_ori):
        # f_2^{-1}(z)
        z_ori = self.enc_nlr(z_ori)
        z = self.enc_nlr(z)
        # masked nonlinear z
        z_new = self.mask(z)
        z_new = z_new + self.get_eps(z_ori)
        return self.prior_nlr(z_new)

    def forward(self, eps_mu=None, eps_sigma=None, z=None):
        if eps_mu is not None and eps_sigma is not None and z is None:
            # (I-A.t)^{-1}*eps
            z_mu, z_sigma = self.inv_cal(eps_mu, eps_sigma)  # n x d
            # nonlinear transform
            return self.prior_nlr(z_mu, z_sigma)
        else:
            # f_2^{-1}(z)
            z = self.enc_nlr(z)
            # mask z
            z_new = self.mask(z)  # new f_2^{-1}(z) (without noise)

            return z_new, z