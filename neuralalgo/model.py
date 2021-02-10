import torch.nn as nn
import torch
import torch.nn.functional as F
from neuralalgo.common.consts import DEVICE, NONLINEARITIES
from neuralalgo.common.utils import weights_init


class QNet(nn.Module):

    def __init__(self, args, hidden_dims_eigs, activation='relu', bias_init='zero'):
        super(QNet, self).__init__()

        self.d = args.d
        self.dim_in = args.dim_in
        self.mu = args.mu
        self.L = args.L
        self.temp = args.temperature
        if activation != 'none':
            self.act_fcn = NONLINEARITIES[activation]

        # generate diagonals
        layers = []
        hidden_dims_eigs = tuple(map(int, hidden_dims_eigs.split("-")))
        prev_size = args.dim_in
        for h in hidden_dims_eigs:
            if h>0:
                layers.append(nn.Linear(prev_size, h, bias=True))
                prev_size = h
        layers.append(nn.Linear(prev_size, self.d-2, bias=True))
        self.layers_w = nn.ModuleList(layers)

        weights_init(self, bias=bias_init)

    def forward(self, u):
        batch = u.shape[0]

        # generate diagonals
        for l, layer in enumerate(self.layers_w):
            if l == 0:
                e = layer(u)
            else:
                e = layer(e)
            if l + 1 < len(self.layers_w):
                e = self.act_fcn(e)
        # shift to [mu, L]
        e = F.softmax(e, dim=-1)
        e = e * (self.L - self.mu) + self.mu

        # concat with mu and L
        eigen1 = torch.ones([batch, 1]).to(DEVICE) * self.L
        eigend = torch.ones([batch, 1]).to(DEVICE) * self.mu
        eigens = torch.cat([eigen1.detach(), e, eigend.detach()], dim=-1)

        return eigens


def qeq(q, e):
    batch, d = e.shape
    qe = q * e.view(batch, 1, d)
    w = torch.matmul(qe, q.transpose(-1, -2))
    return w


def qeq_inv(q, e):
    batch, d = e.shape
    e_inv = 1 / e
    qe_inv = q * e_inv.view(batch, 1, d)
    w_inv = torch.matmul(qe_inv, q.transpose(-1, -2))
    return w_inv
