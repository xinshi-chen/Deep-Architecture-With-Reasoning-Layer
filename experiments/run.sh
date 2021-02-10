#!/bin/bash
echo "script started"

algo_type=gd
phase=pretrain

opt=$1
lr=$2
k=$3
eig_hidden_dims=$4
seed_train=$5

num_train=100
epoch=50000
num_test=10000

seed=2
d=5
dim_in=10
L=1
mu=0.1
init_s=1.0

filename=results_el

dc=1e-4
batch=$((num_train / 2))

subdir=k-${k}-train-${num_train}-test-${num_test}-d-${d}-L-${L}-mu-${mu}-lr-${lr}-dc-${dc}-opt-$opt-e-${eig_hidden_dims}-inits-${init_s}-batch-${batch}-s-${seed_train}
echo "$subdir"
save_dir=../saved_loss/${phase}/${algo_type}/$subdir
algo_model_dump=$save_dir/best_train_algo.dump
e_model_dump=$save_dir/best_train_e.dump
pretrain_e_model_dump=$save_dir/best_pretrain_e.dump

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python3 main.py \
    -algo_type $algo_type \
    -seed $seed \
    -seed_train $seed_train \
    -d $d \
    -eig_hidden_dims $eig_hidden_dims \
    -batch_size $batch \
    -save_dir $save_dir \
    -dim_in $dim_in \
    -learning_rate $lr\
    -weight_decay $dc \
    -num_epochs $epoch \
    -k $k \
    -phase $phase \
    -L $L \
    -mu $mu \
    -num_train $num_train \
    -num_test $num_test \
    -algo_model_dump $algo_model_dump \
    -e_model_dump $e_model_dump \
    -pretrain_e_model_dump $pretrain_e_model_dump \
    -optimizer $opt \
    -init_s $init_s \
    -filename $filename
