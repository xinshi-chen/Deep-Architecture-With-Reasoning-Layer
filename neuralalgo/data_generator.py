from neuralalgo.common.consts import DEVICE
from neuralalgo.common.utils import batch_orthogonal
from neuralalgo.model import qeq_inv
import torch
import numpy as np
import numpy.linalg as la
from scipy.stats import ortho_group


class QuaSPDataset(object):
    def __init__(self, args, true_energy, population=10000, train=1000):
        """
        :param partition_sizes:
        :param d:
        """
        self.d = args.d
        self.dim_in = args.dim_in
        self.u_inf = args.u_inf
        self.u_sup = args.u_sup
        self.b_inf = args.b_inf
        self.b_sup = args.b_sup
        self.static_data = dict()
        self.energy = true_energy
        self.energy.eval()
        self.num_test = population
        self.num_train = train
        u, q, b, x = self.get_samples(population)
        # find label
        self.static_data['test'] = (u, q, b, x)
        self.static_data['train'] = (u[:train], q[:train], b[:train], x[:train])

    def resample_train(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        perms = torch.randperm(self.num_test, generator=g)[:self.num_train].to(DEVICE)
        u, q, b, x = self.static_data['test']
        self.static_data['train'] = u[perms, :], q[perms, :], b[perms, :], x[perms, :]

    def get_samples(self, size):

        u, q, b = self.get_u_q_b(size)
        # label
        with torch.no_grad():
            e = self.energy.forward(u)
            w_inv = qeq_inv(q, e)
            x = torch.einsum('bij,bj->bi', w_inv, b)

        return u, q, b, x.detach()

    def get_u_q_b(self, size):
        # u
        u = torch.rand([size, self.dim_in]).to(DEVICE) * (self.u_sup - self.u_inf) + self.u_inf
        b = torch.rand([size, self.d]).to(DEVICE) * (self.b_sup - self.b_inf) + self.b_inf

        # q -> random orthogonal matrix
        orth_q = torch.tensor(batch_orthogonal(size, self.d)).float().to(DEVICE)

        return u, orth_q, b

    def load_data(self, batch_size, phase, auto_reset=True, shuffle=True):

        assert phase in self.static_data

        u, q, b, x = self.static_data[phase]
        data_size = u.shape[0]
        while True:
            if shuffle:
                perms = torch.randperm(data_size)
                u = u[perms, :]
                q = q[perms, :]
                b = b[perms, :]
                x = x[perms, :]
            for pos in range(0, data_size, batch_size):
                if pos + batch_size > data_size:  # the last mini-batch has fewer samples
                    if auto_reset:  # no need to use this last mini-batch
                        break
                    else:
                        num_samples = data_size - pos
                else:
                    num_samples = batch_size

                yield u[pos : pos + num_samples, :], q[pos : pos + num_samples, :], b[pos : pos + num_samples, :], x[pos : pos + num_samples, :]
            if not auto_reset:
                break

