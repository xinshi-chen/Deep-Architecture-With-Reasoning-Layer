import torch
from tqdm import tqdm
from neuralalgo.common.utils import l2
from neuralalgo.model import qeq
import math
import numpy as np


class QuaSPTrainer:
    def __init__(self, args, data_loader, algo_net, e_net, optimizer, phase='train', dump=True):
        self.args = args
        self.data_loader = data_loader
        self.train_data_loader = data_loader.load_data(batch_size=args.batch_size,
                                                       phase='train')
        self.algo_net = algo_net
        self.e_net = e_net
        if isinstance(optimizer, list):
            self.opt = optimizer
        else:
            self.opt = [optimizer]
        self.data_size = args.num_train
        self.batch_size = args.batch_size
        self.algo_type = args.algo_type
        self.epochs = args.num_epochs
        self.algo_model_dump = args.algo_model_dump

        if phase == 'pretrain':
            self.e_model_dump = args.pretrain_e_model_dump
        else:
            self.e_model_dump = args.e_model_dump
        self.phase = phase

        self.dump = dump

    def _train_epoch(self, epoch, progress_bar, dsc):
        self.algo_net.train()
        self.e_net.train()
        total_loss = 0.0
        iterations = len(range(0, self.data_size, self.batch_size))
        for it in range(iterations):
            u, q, b, x_star = next(self.train_data_loader)
            # forward
            for i in range(len(self.opt)):
                self.opt[i].zero_grad()
            e = self.e_net(u)
            w = qeq(q, e)
            x = self.algo_forward(self.algo_net, w, b, self.algo_type)
            # loss
            loss_batch = l2(x, x_star.detach())
            loss = loss_batch.mean()
            # backward
            loss.backward()

            for i in range(len(self.opt)):
                self.opt[i].step()

            total_loss += loss.item()

            progress_bar.set_description('epoch %.2f, tr err %.3f ' % (epoch + float(it + 1) / iterations, loss.item()) + dsc)
            if math.isnan(float(total_loss)):
                break

        log = {
            'epoch': epoch,
            'avg loss': total_loss / iterations
        }

        return log

    def train(self):
        """
        training logic
        :return:
        """
        # fix algo net
        if self.phase == 'pretrain':
            for param in self.algo_net.parameters():
                param.requires_grad = False

        progress_bar = tqdm(range(self.epochs))
        dsc = ''

        best_loss = math.inf
        gap = 0.0
        best_s = 0.0
        best_w_error_train = 0.0
        for epoch in progress_bar:
            epoch_log = self._train_epoch(epoch, progress_bar, dsc)
            # compute train error
            train_loss, w_error_train = self.eval(self.algo_net, self.e_net, self.data_loader, 'train', self.algo_type, energy_error=True)
            if self.algo_type != 'rnn' and self.algo_type != 'mlp_rnn' and self.algo_type != 'opt':
                ss = self.algo_net.s.item()

            # save best model on train
            if train_loss < best_loss:
                best_loss = train_loss
                best_s = ss
                best_w_error_train = w_error_train
                if self.dump:
                    torch.save(self.e_net.state_dict(), self.e_model_dump)
                    if self.phase != 'pretrain':
                        torch.save(self.algo_net.state_dict(), self.algo_model_dump)

                test_loss, w_error_test = self.eval(self.algo_net, self.e_net, self.data_loader, 'test', self.algo_type, energy_error=True)
                gap = test_loss - train_loss

            dsc = 'train: %.3f, gap: %.3f, bs: %.3f, s:%.3f' % (best_loss, gap, best_s, ss)

        # free algo net
        for param in self.algo_net.parameters():
            param.requires_grad = True

        return best_loss, gap, best_s, best_w_error_train, w_error_test

    @staticmethod
    def eval(algo_net, e_net, data_loader, partition, algo_type, energy_error=False):
        """
        compute the loss on training set or test set
        :param partition:
        :return:
        """
        algo_net.eval()
        e_net.eval()
        static_loader = data_loader.load_data(batch_size=1000,
                                              phase=partition,
                                              auto_reset=False,
                                              shuffle=False)
        total_loss = 0.0
        size = 0
        # e_error = 0.0
        # w_inv_b_error = 0.0
        w_error_sum = 0.0
        for i, data in enumerate(static_loader):
            u, q, b, x_star = data
            size += u.shape[0]
            with torch.no_grad():
                # forward
                e = e_net(u)
                w = qeq(q, e)
                x = QuaSPTrainer.algo_forward(algo_net, w, b, algo_type)
                # loss
                loss = l2(x, x_star).sum()

                if energy_error:
                    # w, b error
                    e_true = data_loader.energy(u)
                    w_true = qeq(q, e_true)
                    w_diff = w_true - w
                    # matrix 2 norm
                    w_error = np.linalg.norm(w_diff.cpu().numpy(), ord=2, axis=(1, 2))
                    w_error_sum += w_error.sum()

            total_loss += loss.item()

        if energy_error:
            return total_loss / size, w_error_sum / size
        return total_loss / size

    @staticmethod
    def algo_forward(network, w, b, algo_type):
        if algo_type == 'nag':
            x, _ = network(w, b)
        else:
            x = network(w, b)
        return x
