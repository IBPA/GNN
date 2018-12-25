#!/usr/bin/env python3
# ./train.py [--dataset=]dataset.csv [--trn=net.tsv] [--save-model-dir=model_dir]

import argparse
import subprocess

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Build and train a GNN model using dataset provided.')
    parser.add_argument('--dataset', metavar='dataset.csv', type=str,
                        required=True, dest="dataset",
                        help='Gene expression profiles.')

    parser.add_argument('--trn', metavar='net.tsv', type=str,
                        required=False, dest="net",
                        help='Gene regulatory network.') 

    parser.add_argument('--save-model-dir', metavar='./model_dir', type=str,
                        required=False, dest="model_dir", default="./model_dir",
                        help='Directory to save trained GNN model.')

    return parser

def run_net_inference(dataset_filename, model_dir):
    # 1) prep genie3 inputs

    # 2) run genie3

    # 3) return net.tsv path

command_args = get_arg_parser().parse_args()

# 1) generate net.tsv if not given (GENIE3)
    net_filename = run_net_inference(command_args['dataset'], command_args['model_dir'])

# 2) generate net.dep and identify MR genes (save into "model_dir/MR_genes.csv")
    #dep_filename = gen_dep_file(net_filename, command_args['model_dir'])

# 3) construct training dataset and ge_ranges.csv
    #prep_data(command_args['dataset'], dep_filename, command_args['model_dir'])

# 4) train GNN model and save into "model_dir/gnn.model"
    #train_model(ommand_args['model_dir'])
