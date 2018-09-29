#!/bin/bash

function divide_multi(){
   base_dir=$1
   for i in $(seq 1 10)
   do
      new_dir=${base_dir}/e${i}/
      mkdir -p $new_dir
      ../prep/run_divide_data.py $base_dir/../input/ $new_dir/d_eval $new_dir/d_net
   done
}

function GENIE3_multi(){
   base_dir=$1
   for i in ${base_dir}/e[0-9]*/d_net
   do
      sbatch -p low -c 24 -n 1 -N 1 -t 300 --job-name=$i ../prep/run_GENIE3.R $i
   done
}

function generate_multi(){
   base_dir=$1
   for e_curr in ${base_dir}/e[0-9]*/
   do
      ../prep/run_generate_module_data.py $e_curr/d_net/edges_inferred.tsv $e_curr/d_eval/data_all_unique.tsv $e_curr/modules/ 1
   done
}

function aggregate_multi_size(){
   base_dir=$1
   eid=$2
   net_size=$3
   modules_dir=$4

   target_dir=$modules_dir/size-${net_size}/top_edges-${eid}_gnw_data/
   src_dir=$base_dir/e${eid}/modules/size-${net_size}/top_edges-1_gnw_data/
   mkdir -p $target_dir
   cp -a $src_dir/* $target_dir
   cp $base_dir/e${eid}/modules/ge_range.csv $target_dir
}

function aggregate_multi(){
   base_dir=$1
   modules_dir=$base_dir/modules_aggr/
   mkdir -p $modules_dir

   for i in $(seq 1 10)
   do
      aggregate_multi_size $base_dir $i 10 $modules_dir
      for net_size in $(seq 100 100 1000)
      do
         aggregate_multi_size $base_dir $i $net_size $modules_dir
      done
   done
}

# 0) Under data/dream5_ecoli/expr1/
base_dir="../data/dream5_ecoli/expr1/"
mkdir -p $base_dir

# 1) for i=1,10: divide data (dream5_ecoli/input) into e[i]/d_net and e[i]/d_eval
#divide_multi $base_dir

# 3) for i=1,10: sbatch ./prep/run_GENIE3.R e[i]/d_net/
#GENIE3_multi $base_dir

# 4) for i=1,10: ./prep/run_generate_module_data.py e[i]/d_net/edges_inferred.tsv e[i]/d_eval/data_all_unique.tsv e[i]/modules/
#generate_multi $base_dir

# 5) ./prep/run_aggregate_multiple_modules.py 
aggregate_multi $base_dir
