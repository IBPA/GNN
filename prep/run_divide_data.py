#!/usr/bin/env python3

# Run Command: ./script_name input_dir output_dir1 output_dir2
# Description: Divide GE dataset of input_dir into two datasets output_dir1, output_dir2.
# Requirements: input_dir should have 3 GE files in DREAM5 challange format (data.tsv, gene_names.tsv, experiments.tsv and tf_names.tsv)
# Summary:
# 0) Load data
# 1) ensure unique condidtions (use mean for replicates)  
# 2) perform clusterning (e.g. num_clusters=27 )
# 3) build dataset in output_dir1 using one sample from each cluster
# 4) build dataset in output_dir2 using the remaining samples

import pandas as pd
import os, sys
import numpy as np
import re
from sklearn.cluster import AgglomerativeClustering
import random
import shutil

########## Functions_Begin ##########
def get_expr_ko_info(line):
    parts = line.split(' D ')
    rKOs = []

    for i in range(1, len(parts)):
        curr = parts[i]
        gene_name = curr.split(' ')[0]
        rKOs.append(gene_name)

    return '/'.join(rKOs)

def get_expr_info(filename):
    info_all = []
    counter = 0
    with open(filename, "r") as bf:
        for line in bf:
            counter += 1
            description = re.sub(" r[0-9]$", "", line).replace("\n", "")
            ko_info = get_expr_ko_info(description)
            info_all.append(['e{0}'.format(counter), ko_info, description])

    df = pd.DataFrame(info_all, columns=['Name', 'Genotype', 'Description'])
    
    return df

def get_data(filename):
    df = pd.read_csv(filename, sep='\t', header=None)
    return df.T

def get_gene_names(filename):
    df = pd.read_csv(filename, sep='\t', header=None)
    return df[0].values

def fu_geom_aggr_mean(x):
    x_ge = x.iloc[:,3:].values
    df_val = pd.DataFrame(np.array(x_ge.mean(axis=0))[np.newaxis]).reset_index(drop=True)
    df_f = pd.DataFrame(x.iloc[0:1,0:3]).reset_index(drop=True)

    df_r = pd.concat([df_f, df_val], axis=1, ignore_index=True)
    df_r.columns = x.columns

    return df_r

def get_aggr(df):
    df_aggr = df.groupby('Description').apply(fu_geom_aggr_mean)

    return df_aggr

def get_samples(df, n_clusters):
    df_ge = df.iloc[:, 3:]
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(df_ge)

    labels_idx = {}
    for i in range(0, len(model.labels_)):
        curr_label = model.labels_[i]
        if curr_label in labels_idx:
            labels_idx[curr_label].append(i)
        else:
            labels_idx[curr_label] = [i]

    idx_picked = []
    idx_remained = []
    for key, value in labels_idx.items():
        rand_int = np.random.randint(0,len(value))
        idx_picked.append(value[rand_int])
        idx_remain_curr = value
        del idx_remain_curr[rand_int]
        idx_remained.extend(idx_remain_curr)
    
    return idx_picked, idx_remained

def save_selected(input_dir, output_dir, df):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_data_file = "{!s}/data_all_unique.tsv".format(output_dir)
    df.to_csv(output_data_file, sep='\t', index=False)
    print('saved to: {!s}'.format(output_data_file))

    shutil.copy2(input_dir['tf_names'], output_dir)
    shutil.copy2(input_dir['gene_names'], output_dir)
    df_ge_t = df.iloc[:,3:].T

    output_data_file = "{!s}/data.tsv".format(output_dir)
    df_ge_t.to_csv(output_data_file, sep='\t', index=False, header=False)
    print('saved to: {!s}'.format(output_data_file))

    output_data_file = "{!s}/experiments.tsv".format(output_dir)
    df.iloc[:,2:2].to_csv(output_data_file, sep='\t', index=False, header=False)
    print('saved to: {!s}'.format(output_data_file))





########## Functions_End ##########

# 0) Load data
input_dir = sys.argv[1] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/input/'.format(os.environ['HOME'])
output_dir1 = sys.argv[2] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/s1/'.format(os.environ['HOME'])
output_dir2 = sys.argv[3] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/s2/'.format(os.environ['HOME'])
input_files = {'data': '{!s}/data.tsv'.format(input_dir),
               'gene_names': '{!s}/gene_names.tsv'.format(input_dir),
               'experiments': '{!s}/experiments.tsv'.format(input_dir),
                'tf_names': '{!s}/tf_names.tsv'.format(input_dir)}
expr_info_df = get_expr_info(input_files['experiments'])
data_df = get_data(input_files['data'] )
gene_names = get_gene_names(input_files['gene_names'])
data_df.columns = gene_names
df = pd.concat([expr_info_df,data_df], axis=1)

# 1) Ensure unique condidtions (use mean for replicates)  
df = get_aggr(df)
output_data_file = "{!s}/data_all_unique.tsv".format(input_dir)
df.to_csv(output_data_file, sep='\t', index=False)
print('saved to: {!s}'.format(output_data_file))

# 2) perform clusterning (e.g. n_clusters=27 )
n_clusters=27
idx_picked, idx_remained = get_samples(df, n_clusters)

# 3) build dataset in output_dir1 using one sample from each cluster
save_selected(input_files, output_dir1, df.iloc[idx_picked])

# 4) build dataset in output_dir2 using the remaining samples
save_selected(input_files, output_dir2, df.iloc[idx_remained])