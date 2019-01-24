#!/usr/bin/env python3
""" Generate hyper-parameters for MLP and save 'in hparam_kmlp.json'
   Arg1: base directory name.
"""

import os, sys, csv
import json
import lib.keras_util as keras_util

base_dir = sys.argv[1] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/expr1/modules_aggr/'.format(os.environ['HOME'])
h_params = keras_util.getMLPHyperParameters(min_nodes=5, max_nodes=50, min_alpha=0, max_alpha=5, num_combinations=50)
hyper_param_filename = '{!s}/hparam_kmlp.json'.format(base_dir)
with open(hyper_param_filename, 'w') as fHandle:
    json.dump(h_params, fHandle)
    print("saved to: {!s}".format(hyper_param_filename))