from neuralalgo.data_generator import QuaSPDataset
from neuralalgo.common.cmd_args import cmd_args
from neuralalgo.common.consts import DEVICE
from neuralalgo.model_algos import GD, NAG, MLPRNN, OptAlgo
from neuralalgo.model import QNet
from neuralalgo.trainer import QuaSPTrainer
import random
import numpy as np
import torch
import torch.optim as optim
import pickle


def append_results(results, args, train_loss, test_loss, s, w_error_train, w_error_test):

    results.setdefault('algo', []).append(args.algo_type)
    results.setdefault('s', []).append(s)
    results.setdefault('k', []).append(args.k)
    results.setdefault('train_loss', []).append(train_loss)
    results.setdefault('test_loss', []).append(test_loss)
    results.setdefault('gap', []).append(test_loss-train_loss)
    results.setdefault('w_err_train', []).append(w_error_train)
    results.setdefault('w_err_test', []).append(w_error_test)
    results.setdefault('mu', []).append(args.mu)
    results.setdefault('L', []).append(args.L)
    results.setdefault('num_train', []).append(args.num_train)
    results.setdefault('num_test', []).append(args.num_test)
    results.setdefault('epoch', []).append(args.num_epochs)
    results.setdefault('optimizer', []).append(args.optimizer)
    results.setdefault('lr', []).append(args.learning_rate)
    results.setdefault('dc', []).append(args.weight_decay)
    results.setdefault('dim', []).append(args.eig_hidden_dims)
    results.setdefault('withq', []).append(args.withq)
    results.setdefault('seed', []).append(args.seed_train)
    results.setdefault('dump', []).append(args.e_model_dump)

    return results


def save_to_pkl(args, train_err, gen_gap, s, w_error_train, w_error_test, filename='results'):

    train_loss = train_err
    test_loss = train_err + gen_gap

    print('train:%.3f,gap:%.3f,s:%.3f' % (train_loss, test_loss-train_loss, s))

    this_result = append_results({}, args, train_loss, test_loss, s, w_error_train, w_error_test)
    save_result_filename = filename + '.pkl'
    with open(save_result_filename, 'ab') as handle:
        pickle.dump(this_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    best_result_filename = filename + '_best.pkl'
    try:
        with open(best_result_filename, 'rb') as handle:
            best_results = pickle.load(handle)
    except IOError:
        best_results = {}

    # replace best result
    if args.eig_hidden_dims not in best_results:
        best_results[args.eig_hidden_dims] = {}
    if args.algo_type not in best_results[args.eig_hidden_dims]:
        best_results[args.eig_hidden_dims][args.algo_type] = {}
    if args.k not in best_results[args.eig_hidden_dims][args.algo_type]:
        best_results[args.eig_hidden_dims][args.algo_type][args.k] = {}
    if args.seed_train not in best_results[args.eig_hidden_dims][args.algo_type][args.k]:
        # save if this is the first result
        best_results[args.eig_hidden_dims][args.algo_type][args.k][args.seed_train] = this_result
        with open(best_result_filename, 'wb') as handle:
            pickle.dump(best_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # compare training loss
        current_best_dict = best_results[args.eig_hidden_dims][args.algo_type][args.k][args.seed_train]
        if train_loss < current_best_dict['train_loss'][0]:
            # replace if better
            best_results[args.eig_hidden_dims][args.algo_type][args.k][args.seed_train] = this_result
            with open(best_result_filename, 'wb') as handle:
                pickle.dump(best_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    
    # load groundtruth energy
    true_energy = QNet(args=cmd_args,
                       hidden_dims_eigs='3',
                       activation='relu').to(DEVICE)
    true_energy.load_state_dict(torch.load(cmd_args.true_energy_dump))

    # initialize the train, test database
    db = QuaSPDataset(args=cmd_args,
                      true_energy=true_energy,
                      population=cmd_args.num_test,
                      train=cmd_args.num_train)

    # resample training data
    db.resample_train(cmd_args.seed_train)

    # initialize the algo network
    if cmd_args.algo_type == 'gd':
        algo = GD(k=cmd_args.k, init_s=cmd_args.init_s).to(DEVICE)
    elif cmd_args.algo_type == 'nag':
        algo = NAG(k=cmd_args.k, mu=cmd_args.mu, init_s=cmd_args.init_s).to(DEVICE)
    elif cmd_args.algo_type == 'opt':
        algo = OptAlgo()
    else:
        assert cmd_args.algo_type == 'mlp_rnn'
        algo = MLPRNN(k=cmd_args.k, hidden_dim=cmd_args.mlp_hidden_dims, d=cmd_args.d, activation=cmd_args.activation).to(DEVICE)
        algo.init_as_gd()

    # initialize the energy network
    energy = QNet(args=cmd_args,
                  hidden_dims_eigs=cmd_args.eig_hidden_dims,
                  activation=cmd_args.activation).to(DEVICE)

    # -> fix algo, learn energy

    if cmd_args.optimizer == 'sgd':
        optimizer = optim.SGD(energy.parameters(),
                              lr=cmd_args.learning_rate,
                              weight_decay=cmd_args.weight_decay)
    else:
        assert cmd_args.optimizer == 'adam'
        optimizer = optim.Adam(energy.parameters(),
                               lr=cmd_args.learning_rate,
                               weight_decay=cmd_args.weight_decay)

    trainer = QuaSPTrainer(args=cmd_args,
                           data_loader=db,
                           algo_net=algo,
                           e_net=energy,
                           optimizer=optimizer,
                           phase=cmd_args.phase,
                           dump=False)

    # reset seed before train
    random.seed(cmd_args.seed2)
    np.random.seed(cmd_args.seed2)
    torch.manual_seed(cmd_args.seed2)

    best_loss, gap, best_s, w_error_train, w_error_test = trainer.train()
    save_to_pkl(cmd_args, best_loss, gap, best_s, w_error_train, w_error_test, cmd_args.filename)
