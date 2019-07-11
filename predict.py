#!/usr/bin/env python3
""" Predict GE using trained GNN model """

import argparse
import subprocess
import os, sys
import numpy as np
import pandas as pd
_script_dir = os.path.dirname(os.path.realpath(__file__))

def get_arg_parser():
    """ Build command line parser
    Returns:
        command line parser
    """

    parser = argparse.ArgumentParser(description='Predict Gene Expression using a trained GNN model.')
    parser.add_argument('--input', metavar='gnn_input.csv', type=str,
                        required=True, dest="gnn_input",
                        help='Knockout info and Master Regulator expressions.')

    parser.add_argument('--output', metavar='gnn_pred.csv', type=str,
                        required=True, dest="output_filename",
                        help='Predicted gene expression values.') 

    parser.add_argument('--load-model-dir', metavar='./model_dir', type=str,
                        required=False, dest="model_dir", default="./model_dir",
                        help='Directory to load trained GNN model.')

    return parser


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

def get_mr_nmr(dep_filename):
    mr_list = []
    nmr_list = []

    with open(dep_filename, "r") as f:
        line = f.readline()
        while line:
            parts = line.strip().split(":")
            if (len(parts[1]) < 1):
                mr_list.append(parts[0])
            else:
                nmr_list.append(parts[0])
            line = f.readline()

    return mr_list, nmr_list

def prep_data(input_data_filename, model_dir):
    df = pd.read_csv(input_data_filename)
    input_files = {'mr': "{!s}/input_MR.tsv".format(model_dir),
                   'ko': "{!s}/input_KO.tsv".format(model_dir),
                   'dep': "{!s}/net.dep".format(model_dir) }

    mr_list, nmr_list = get_mr_nmr(input_files['dep'])

    df[mr_list].to_csv(input_files['mr'], sep="\t", index=False)
    
    ko_bin_df = get_ko_binary_df(df['KO'], nmr_list)
    ko_bin_df.to_csv(input_files['ko'], sep="\t", index=False)

    return input_files

def pred_GE(model_dir, output_filename):
    train_script = "th {!s}/predict_GNN.lua {!s} {!s}".format(_script_dir, model_dir, output_filename)
    subprocess.call([train_script], shell=True)

command_args = get_arg_parser().parse_args()

prep_data(command_args.gnn_input, command_args.model_dir)

pred_GE(command_args.model_dir, command_args.output_filename)
