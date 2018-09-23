#!/usr/bin/env python3

# Run Command: ./script_name input_edges_file input_data_file output_dir
# Description: Extracting network modules and corresponding GE data for prediction benchmark
# Requirements: input_edges_file: should contain rows witt "src dst weight" records. input_data_file: should contain GE daata
# Summary:
# 1) Generate TRN modules
# 2) Generate GE data for each module

import os, sys, csv
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import net_gen
import net_data_gen_real
import stratify

########## Function_Begin ##########

def get_top_edges(edges_filepath, num_nodes, frac_edges):
    edges = []
    nodes = set()
    with open(edges_filepath, 'r') as f:
        for line in f.readlines():
            src, dst, sign = line.strip().split('\t')
            edges.append([src, dst, "-"])
            nodes.add(src)
            nodes.add(dst)

            if len(nodes) > num_nodes and len(edges) > num_nodes*frac_edges:
                break
    
    return edges

def save_edges_tsv(list_top_edges, edges_filepath):
    with open(edges_filepath, 'w') as f:
        wr = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        wr.writerows(list_top_edges)

def save_mr_score(input_edges_file, mr_score_filepath):
    df = pd.read_csv(input_edges_file, sep="\t", header=None)
    df.columns = ['src', 'dst', 'weight']

    df_avg_out = pd.DataFrame(df.groupby(['src'], as_index=False).mean())

    df_avg_out.to_csv(mr_score_filepath, sep='\t', index=False)
    print("saved to {!s}".format(mr_score_filepath))

def save_min_max(df, output_filepath):
    df_minmax = pd.concat([df.min(), df.max()], axis=1)
    df_minmax.columns = ['min', 'max']
    df_minmax.index.name = 'gene'
    df_minmax.to_csv(output_filepath, sep=',')
    print("saved: {!s}".format(output_filepath))

def generate_data_prep(input_edges_file, df, output_dir):
    gen_filenames = {'mr_score': "{!s}/mr_score.tsv".format(output_dir), 
                    'ge_range': "{!s}/ge_range.csv".format(output_dir)}
    
    save_mr_score(input_edges_file, gen_filenames['mr_score'])
    save_min_max(df.iloc[:,3:], gen_filenames['ge_range'])

    return gen_filenames

def generate_modules(edges_file, num_nodes, frac_edges, output_dir, gnw_path):
    list_top_edges = get_top_edges(edges_file, num_nodes, frac_edges)
    curr_edges_filepath = '{!s}/top_edges.tsv'.format(output_dir)
    save_edges_tsv(list_top_edges, curr_edges_filepath)

    module_files = net_gen.run(gnw_path=gnw_path,
                        min_netsize=num_nodes,
                        max_netsize=num_nodes+1,
                        stepsize=100,
                        num_nets_per_size=10,
                        main_netfile_path=curr_edges_filepath,
                        output_prefix = "{!s}/size-".format(output_dir))

    return module_files

def generate_data(input_edges_file, input_data_file, module_files, output_dir):
    df = pd.read_csv(input_data_file, sep="\t")
    gen_filenames = generate_data_prep(input_edges_file, df, output_dir)
    for net_file in module_files:
        gen_data_dir = net_data_gen_real.generate(gen_filenames, df, net_file)
        stratify.generate(gen_data_dir)

########## Function_End ##########

input_edges_file = sys.argv[1] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/s2/edges_inferred.tsv'.format(os.environ['HOME'])
input_data_file = sys.argv[2] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/s1/data_all_unique.tsv'.format(os.environ['HOME'])
output_dir = sys.argv[3] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/modules/'.format(os.environ['HOME'])
gnw_path = '{!s}/mygithub/gnw/gnw-3.1.2b.jar'.format(os.environ['HOME'])

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

frac_edges=1.2

# 1) Generate TRN modules
num_nodes=10
module_files = generate_modules(input_edges_file, num_nodes, frac_edges, output_dir, gnw_path)
for num_nodes in range(100, 1100, 100):
    module_files_curr = generate_modules(input_edges_file, num_nodes, frac_edges, output_dir, gnw_path)
    module_files.extend(module_files_curr)

# 2) Generate GE data for each module
generate_data(input_edges_file, input_data_file, module_files, output_dir)