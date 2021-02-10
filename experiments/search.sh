#!/bin/bash

d=32
for k in 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30; do
    for opt in adam sgd; do
        for lr in 1e-2 1e-3 5e-4 1e-4; do
            for((s=100;s<=119;s++)) do
                bash run_gd.sh $opt $lr $k $d $s
            done
        done
    done
done
