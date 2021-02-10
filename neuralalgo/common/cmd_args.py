import argparse
import os

cmd_opt = argparse.ArgumentParser(description='')

cmd_opt.add_argument('-f_model_dump', type=str, default='./scratch/f_model.dump')
cmd_opt.add_argument('-g_model_dump', type=str, default='./scratch/g_model.dump')
cmd_opt.add_argument('-dump', type=eval, default=False)

cmd_opt.add_argument('-f_hidden_dim', type=int, default=5)
cmd_opt.add_argument('-f_num_layer', type=int, default=3)

cmd_opt.add_argument('-g_hidden_dim', type=int, default=5)
cmd_opt.add_argument('-g_num_layer', type=int, default=3)




cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-save_dir', type=str, default='./scratch', help='save folder')
cmd_opt.add_argument('-seed', type=int, default=2, help='seed')
cmd_opt.add_argument('-seed2', type=int, default=10086, help='reset seed for training')
cmd_opt.add_argument('-seed3', type=int, default=999983, help='reset seed for training')
cmd_opt.add_argument('-seed_train', type=int, default=999983, help='reset seed for training')

cmd_opt.add_argument('-phase', type=str, default='train', help='training or testing phase')


# hyperparameters for training and testing
cmd_opt.add_argument('-num_test', type=int, default=5000, help='number of test samples')
cmd_opt.add_argument('-num_train', type=int, default=100, help='number of train samples')
cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-init_s', type=float, default=1e-5, help='initialization for step size s')

cmd_opt.add_argument('-num_epochs', type=int, default=100, help='num epochs')
cmd_opt.add_argument('-save_file', type=str, default='./scratch/saveloss.pkl', help='filename for saving the losses')
cmd_opt.add_argument('-model_dump', type=str, default='./scratch/model.dump', help='best model on training set dump')
cmd_opt.add_argument('-true_energy_dump', type=str, default='./scratch/true_energy_model.dump')

cmd_opt.add_argument('-qua_energy_dump', type=str, default='./scratch/true_qua_energy.dump')
cmd_opt.add_argument('-diag_energy_dump', type=str, default='./scratch/true_diag_energy.dump')
cmd_opt.add_argument('-algo_model_dump', type=str, default='./scratch/algo_model.dump')
cmd_opt.add_argument('-e_model_dump', type=str, default='./scratch/e_model.dump')
cmd_opt.add_argument('-pretrain_e_model_dump', type=str, default='./scratch/pretrain_e_model.dump')

cmd_opt.add_argument('-filename', type=str, default='results_2stage.pkl', help='filename for saving the results')

cmd_opt.add_argument('-optimizer', type=str, default='sgd')


cmd_opt.add_argument('-save_loc', type=str, default=None, help='filename for saving the losses')


# hyperparameters for the Quadratic Optimization problem instance
cmd_opt.add_argument('-d', type=int, default=5, help='dim of x')
cmd_opt.add_argument('-dim_in', type=int, default=10, help='dim of u')

cmd_opt.add_argument('-mu', type=float, default=0.1, help='convexity')
cmd_opt.add_argument('-L', type=float, default=1, help='smoothness')

# hyperparameters for the structured prediction problem instance
cmd_opt.add_argument('-u_inf', type=float, default=-5.0, help='inf of u')
cmd_opt.add_argument('-u_sup', type=float, default=5.0, help='sup of u')
cmd_opt.add_argument('-b_inf', type=float, default=-5.0, help='inf of b')
cmd_opt.add_argument('-b_sup', type=float, default=5.0, help='sup of b')
cmd_opt.add_argument('-withq', type=eval, default=True)

# RNN hyperparameters
cmd_opt.add_argument('-dim_hidden', type=int, default=10, help='hidden dimension in RNN')
cmd_opt.add_argument('-activation', type=str, default='relu', help='activation in RNN')
cmd_opt.add_argument('-mlp_hidden_dims', type=str, default='20-20-20', help='hidden dimension in MLP RNN')

# Energy Net hyperparameters
cmd_opt.add_argument('-w_hidden_dims', type=str, default='20-20-20', help='hidden dimension in matrix generator')
cmd_opt.add_argument('-b_hidden_dims', type=str, default='20-20-20', help='hidden dimension in b generator')
cmd_opt.add_argument('-eig_hidden_dims', type=str, default='20-20-20', help='hidden dimension in eigenvalues generator')
cmd_opt.add_argument('-temperature', type=float, default=4.0, help='temperature for softmax')


# hyperparameters for the LISTA problem instance
cmd_opt.add_argument('-dim_n', type=int, default=1000, help='dim of x')
cmd_opt.add_argument('-dim_m', type=int, default=250, help='dim of y')
cmd_opt.add_argument('-snr', type=float, default=None, help='signal noise ratio')
cmd_opt.add_argument('-sp', type=float, default=0.1, help='sparsity')
cmd_opt.add_argument('-l_inf', type=float, default=1, help='upper bound of |x|_inf')
cmd_opt.add_argument('-con_num', type=float, default=5, help='condition number of matrix A')

# hyperparameters for the algorithms
cmd_opt.add_argument('-depth', type=int, default=20, help='number of iterations in the algorithm')
cmd_opt.add_argument('-k', type=int, default=20, help='number of iterations in the algorithm')
cmd_opt.add_argument('-algo_type', type=str, default='gd', choices=['gd', 'nag', 'rnn', 'mlp_rnn', 'debug'])


cmd_opt.add_argument('-theta_init', type=float, default=None)
cmd_opt.add_argument('-alpha_init', type=float, default=None)


cmd_args = cmd_opt.parse_args()
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
