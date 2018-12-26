#!/usr/bin/env python3.5
""" Build and train GNN model """

import argparse
import subprocess
import os, sys
import lib.graph_util as graph_util
import lib.impact_module as impact_module
import pandas as pd

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


command_args = get_arg_parser().parse_args()

if not os.path.exists(command_args.model_dir):
    os.makedirs(command_args.model_dir)

# 1) generate net.tsv if not given (GENIE3)
if command_args.net == None:
    net_filename = run_net_inference(command_args.dataset, command_args.model_dir)

# 2) generate net.dep and identify MR genes (save into "model_dir/MR_genes.csv")
dep_filename, mr_list, nmr_list = gen_dep_file(net_filename, command_args.model_dir)

# 3) construct training dataset and ge_ranges.csv
#prep_data(command_args['dataset'], dep_filename, command_args['model_dir'])

# 4) train GNN model and save into "model_dir/gnn.model"
#train_model(ommand_args['model_dir'])
