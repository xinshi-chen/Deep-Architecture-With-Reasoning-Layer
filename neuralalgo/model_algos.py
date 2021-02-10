import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from neuralalgo.common.consts import DEVICE, NONLINEARITIES


class OptAlgo(nn.Module):
    def __init__(self):
        super(OptAlgo, self).__init__()

    def forward(self, w, b, w_inv=None):
        if w_inv is None:
            w_inv = torch.inverse(w)
        return torch.einsum('bij,bj->bi', w_inv, b)


class GD(nn.Module):
    def __init__(self, k, init_s=None):
        super(GD, self).__init__()
        self.k = k
        if init_s is None:
            self.s = Parameter(torch.tensor(1e-6).to(DEVICE))
        else:
            self.s = Parameter(torch.tensor(init_s).to(DEVICE))

    def forward(self, w, b, return_all=False, x_init=None):
        if x_init is not None:
            x = x_init.clone()
        else:
            x = torch.zeros(size=b.shape).to(DEVICE)

        if return_all:
            x_t = [x]
            for _ in range(self.k):
                x = self.gradient_decent(self.s, x, w, b)
                x_t.append(x)
            return x_t

        else:
            for _ in range(self.k):
                x = self.gradient_decent(self.s, x, w, b)
            return x

    @staticmethod
    def gradient_decent(s, x, w, b):
        # grad = wx-b
        gradient = torch.einsum('bij,bj->bi', w, x) - b
        # descent
        if len(s.shape) > 0:
            s = s.view(-1, 1)
        x = x - s * gradient
        return x


class GDTrivial(GD):
    def __init__(self, k):
        super(GDTrivial, self).__init__(k)

    def forward(self, w, b, return_all=False, x_init=None):
        x = torch.ones(size=b.shape).to(DEVICE)

        for _ in range(self.k):
            x = self.gradient_decent(self.s, x, w, b)
        return x


class NAG(GD):
    def __init__(self, k, mu=None, init_s=None):
        super(NAG, self).__init__(k, init_s)
        self.mu = mu

    def forward(self, w, b, return_all=False, x_init=None, y_init=None, compute_s=False):

        if self.mu is not None:
            mu = self.mu
        else:
            # estimate the smallest eigenvalues of w
            with torch.no_grad():
                eigs, _ = torch.symeig(w.cpu(), eigenvectors=False)
                mu = F.relu(eigs.min(dim=-1)[0] - 1e-7) + 1e-7  # make sure mu is positive
                mu = mu.to(DEVICE).detach()  # not differentiate through mu

        if x_init is not None:
            x = x_init.clone()
            y = y_init.clone()
        else:
            x = torch.zeros(size=b.shape).to(DEVICE)
            y = torch.zeros(size=b.shape).to(DEVICE)

        if return_all:
            x_t = [x]
            y_t = [y]
            for _ in range(self.k):
                x, y = self.nag(self.s, mu, x, y, w, b)
                x_t.append(x)
                y_t.append(y)
            return x_t, y_t

        else:
            for _ in range(self.k):
                x, y = self.nag(self.s, mu, x, y, w, b)
            return x, y

    @staticmethod
    def nag(s, mu, x, y, w, b):
        if s is None:
            _, eigs, _ = torch.svd(w)
            L = eigs.max(dim=-1)[0]
            s = 1 / L.detach()
            if mu is None:
                mu = F.relu(eigs.min(dim=-1)[0] - 1e-7) + 1e-7
        else:
            if mu is None:
                _, eigs, _ = torch.svd(w)
                mu = F.relu(eigs.min(dim=-1)[0] - 1e-7).detach() + 1e-7

        x_next = GD.gradient_decent(s, y, w, b)
        alpha = (1 - torch.sqrt(mu * s)) / (1 + torch.sqrt(mu * s))
        if len(alpha.shape) > 0:
            alpha = alpha.view(-1, 1)
        y = (1 + alpha) * x_next - alpha * x
        return x_next, y


class VallinaRNN(nn.Module):
    def __init__(self, k, hidden_dim, d, activation='relu'):
        super(VallinaRNN, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.weight_w = Parameter(torch.Tensor(hidden_dim, d))
        self.weight_u = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_v = Parameter(torch.Tensor(d, hidden_dim))
        nn.init.eye_(self.weight_w)
        nn.init.zeros_(self.weight_u)
        nn.init.eye_(self.weight_v)
        if activation != 'none':
            self.act_fcn = NONLINEARITIES[activation]

    def forward(self, w, b, return_all=False, x_init=None, h_init=None):

        if x_init is not None:
            x = x_init.clone()
            h = h_init.clone()
        else:
            x = torch.zeros(size=b.shape).to(DEVICE)
            h = torch.zeros(size=[b.shape[0], self.hidden_dim]).to(DEVICE)

        if return_all:
            x_t = [x]
            for _ in range(self.k):
                x, h = self.cell(x, h, w, b)
                x_t.append(x)
            return x_t
        else:
            for _ in range(self.k):
                x, h = self.cell(x, h, w, b)
            return x

    def cell(self, x, h, w, b):
        # grad = wx-b
        gradient = torch.einsum('bij,bj->bi', w, x) - b
        h = F.linear(h, self.weight_u, bias=None) + F.linear(-gradient, self.weight_w, bias=None)
        if self.activation != 'none':
            h = self.act_fcn(h)
        x = x + F.linear(h, self.weight_v, bias=None)
        return x, h


class MLPRNN(nn.Module):
    def __init__(self, k, hidden_dim, d, activation='relu'):
        super(MLPRNN, self).__init__()
        hidden_dims = tuple(map(int, hidden_dim.split("-")))
        input_dim = 2 * d
        self.activation = activation
        self.d = d
        self.k = k
        self.hidden_dims = hidden_dims
        weight_w = []
        for h in hidden_dims:
            weight_w.append(Parameter(torch.Tensor(h, input_dim)))
            input_dim = h
        self.weight_w = nn.ParameterList(weight_w)

        self.weight_v = Parameter(torch.Tensor(d, input_dim))

        for l in range(len(self.weight_w)):
            nn.init.eye_(self.weight_w[l])
        nn.init.eye_(self.weight_v)
        if activation != 'none':
            self.act_fcn = NONLINEARITIES[activation]

    def forward(self, w, b, return_all=False, x_init=None):

        if x_init is not None:
            x = x_init.clone()
        else:
            x = torch.zeros(size=b.shape).to(DEVICE)

        if return_all:
            x_t = [x]
            for _ in range(self.k):
                x = self.cell(x, w, b)
                x_t.append(x)
            return x_t
        else:
            for _ in range(self.k):
                x = self.cell(x, w, b)
            return x

    def cell(self, x, w, b):
        # grad wx-b
        gradient = torch.einsum('bij,bj->bi', w, x) - b
        h = torch.cat([-gradient, x], dim=-1)
        # MLP
        for l in range(len(self.weight_w)):
            h = F.linear(h, self.weight_w[l], bias=None)
            if self.activation != 'none':
                h = self.act_fcn(h)
        x = F.linear(h, self.weight_v, bias=None)
        return x

    def init_as_gd(self):
        l1 = self.hidden_dims[0]
        l2 = self.hidden_dims[1]
        assert l1 == 2*self.d
        assert l1 == l2
        # w0 = [I, I; -I, -I]
        init_w0 = torch.cat([torch.eye(self.d), torch.eye(self.d)], dim=-1)
        init_w0 = torch.cat([init_w0, -init_w0], dim=0).to(DEVICE)
        self.weight_w[0].data = init_w0
        for l in range(2, len(self.hidden_dims)):
            nn.init.eye_(self.weight_w[l])
        assert self.hidden_dims[-1] == l1
        init_v = torch.cat([torch.eye(self.d), -torch.eye(self.d)], dim=-1).to(DEVICE)
        self.weight_v.data = init_v
