#!/bin/bash

for str_dir in ../data/dream5_ecoli/modules/size-[0-9]*
do
   #sbatch -p low -n 1 -N 1 -t 300 ./aggregate_per_netSize.R $str_dir
   sbatch -A bc5fp1p -p RM-shared --ntasks-per-node 2 -N 1 -t 7:58:00 ../vis/aggregate_per_netSize.R $str_dir
done
