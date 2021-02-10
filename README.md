# Understanding Deep Architectures With Reasoning Layer (NeurIPS 2020)

- Authors: [Xinshi Chen](http://xinshi-chen.com/), [Yufei Zhang](https://www.maths.ox.ac.uk/people/yufei.zhang), [Christoph ](https://www.maths.ox.ac.uk/people/christoph.reisinger), [Le Song](https://www.cc.gatech.edu/~lsong/)

- [Link to paper](https://papers.nips.cc/paper/2020/hash/0d82627e10660af39ea7eb69c3568955-Abstract.html)

- [Link to slides](http://xinshi-chen.com/papers/slides/nips2020-reasoning.pdf)

- [Link to NeurIPS presentation](https://nips.cc/virtual/2020/protected/poster_0d82627e10660af39ea7eb69c3568955.html)

If you found this library useful in your research, please consider citing

```
@article{chen2020understanding,
  title={Understanding Deep Architecture with Reasoning Layer},
  author={Chen, Xinshi and Zhang, Yufei and Reisinger, Christoph and Song, Le},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

# Reproduce Experiments

## Install the module
Please navigate to the root of this repository, and run the following command to install the `neuralalgo` module.
```
pip install -e .
```

## Run the experiments
Navigate to the `experiments/` folder and use the run file `run.sh` to run the experiments. There are severl hyperparameters in this file:
```
% line 7. This can be set as 'sgd' or 'adam', which is the optimizer for training the model.
opt=$1
% line 8. This is the learning rate. In our paper, we search over 1e-2 1e-3 5e-4 1e-4.
lr=$2
% line 9. This is the number of algorithm steps (iterations) that is used in the reasoning module.
k=$3
% line 10. This is the hidden dimension of the MLP in the neural module.
eig_hidden_dims=$4
% line 11. This is the random seed that we use to provide multiple independent instantiations of the experiments.
seed_train=$5
```
