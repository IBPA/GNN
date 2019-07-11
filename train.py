#!/usr/bin/env python3
""" Build and train GNN model """

import argparse
import subprocess
import os, sys
import lib.graph_util as graph_util
import lib.impact_module as impact_module
import pandas as pd
import numpy as np

_script_dir = os.path.dirname(os.path.realpath(__file__))

def get_arg_parser():
    """ Build command line parser
    Returns:
        command line parser
    """

    parser = argparse.ArgumentParser(description='Build and train a GNN model using dataset provided.')
    parser.add_argument('--dataset', metavar='dataset.csv', type=str,
                        required=True, dest="dataset",
                        help='Gene expression profiles.')

    parser.add_argument('--trn', metavar='net.tsv', type=str,
                        required=False, dest="net",
                        help='Gene regulatory network.') 

    parser.add_argument('--output-model-dir', metavar='./model_dir', type=str,
                        required=False, dest="model_dir", default="./model_dir",
                        help='Directory to save trained GNN model.')

    return parser

def run_net_inference(dataset_filename, model_dir):
    """ Wrapper for GENIE3 gene network inference
    Args:
        dataset_filename: filename containing GE values.
        model_dir: directory name to save generated net_GENIE3.tsv file.
    Returns:
        file path for generated network
    """

    net_filename = "{!s}/net_GENIE3.tsv".format(model_dir)
    GENIE3_script_filename = "{!s}/prep/run_GENIE3_new.R".format(_script_dir)
    command = "{!s} -i {!s} -o {!s}".format(GENIE3_script_filename, dataset_filename, net_filename)
    subprocess.call([command], shell=True)

    return net_filename

def get_mr_scores(net_filename):
    df = pd.read_csv(net_filename, sep="\t")
    if (len(df.columns) == 2):
        df['weight'] = float(1.0)
    df.columns = ['src', 'dst', 'weight']
    df_avg_out = pd.DataFrame(df.groupby(['src'], as_index=False).mean())

    return df_avg_out

def gen_dep_file(net_filename, model_dir):
    impact_m = impact_module.OutWImpact(df_input=get_mr_scores(net_filename))
    graph = graph_util.DirGraphReal(net_filename, impact_m, has_header=True)
    mr_list = graph.get_mr()
    nmr_list = graph.get_nmr()
    dep_filename = "{!s}/net.dep".format(model_dir)
    graph.save_as_linear_dep_graph(mr_list, dep_filename)

    return dep_filename, mr_list, nmr_list

def get_ko_vector(ko_str, nmr_list):
    ko_vector = np.ones(len(nmr_list))
    
    if type(ko_str) is str and len(ko_str)>0 :
        ko_str_parts = ko_str.split("&")

        for gname in ko_str_parts:
            if gname in nmr_list:
                gidx = nmr_list.index(gname)
                ko_vector[gidx] = 0
            
    return ko_vector

def get_ko_binary_df(ko_df, nmr_list):
    def get_ko(x):
        return get_ko_vector(x, nmr_list)

    ko_data = ko_df.apply(get_ko)
    df = pd.DataFrame.from_records(ko_data, columns=nmr_list)
    return df

def prep_data(dataset_filename, dep_filename, model_dir, mr_list, nmr_list):
    df = pd.read_csv(dataset_filename)
    train_files = {'mr': "{!s}/train_MR.tsv".format(model_dir),
                   'nmr': "{!s}/train_NMR.tsv".format(model_dir),
                   'ko': "{!s}/train_KO.tsv".format(model_dir) }

    df[mr_list].to_csv(train_files['mr'], sep="\t", index=False)
    df[nmr_list].to_csv(train_files['nmr'], sep="\t", index=False)
    
    ko_bin_df = get_ko_binary_df(df['KO'], nmr_list)
    ko_bin_df.to_csv(train_files['ko'], sep="\t", index=False)

    return train_files

def train_model(model_dir):
    train_script = "th {!s}/train_GNN.lua {!s}".format(_script_dir, model_dir)
    subprocess.call([train_script], shell=True)

command_args = get_arg_parser().parse_args()

if not os.path.exists(command_args.model_dir):
    os.makedirs(command_args.model_dir)

# 1) generate net.tsv if not given (GENIE3)
net_filename = command_args.net
if net_filename == None:
    net_filename = run_net_inference(command_args.dataset, command_args.model_dir)

# 2) generate net.dep and identify MR genes (save into "model_dir/MR_genes.csv")
dep_filename, mr_list, nmr_list = gen_dep_file(net_filename, command_args.model_dir)

# 3) construct training dataset
prep_data(command_args.dataset, dep_filename, command_args.model_dir, mr_list, nmr_list)

# 4) train GNN model and save into "model_dir/gnn.model"
train_model(command_args.model_dir)
