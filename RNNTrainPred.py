""" Train Recurrent Neural Network and test.
   Arg1: base directory name for data files.
   Arg2: file name containing fold information (first row training indexes, second row test indexes)
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import itertools as it
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.regularizers import l2
import lib.data_rw as data_rw
import lib.keras_util as keras_util

def getModel(nDInput, nDOutput, alpha, nDepth):
    model = Sequential()
    model.add(SimpleRNN(units=nDOutput, input_dim= nDInput,
                        kernel_regularizer=l2(alpha), activation='sigmoid'))
    
    return model

def run(data_filenames, fold_filename, alpha, nDepth, ge_range_all):
    fold_filepath = '{!s}/folds/{!s}'.format(dir_path, fold_filename)
    df_train_input, df_train_output, df_test_input, df_test_output = data_rw.loadData(data_filenames, fold_filepath)
    nM = df_train_input.shape[0]

    model = getModel(nDInput = df_train_input.shape[1],
                     nDOutput = df_train_output.shape[1],
                     alpha = alpha,
                     nDepth = nDepth)
    
    if ge_range_all:
        model = keras_util.apply_range(model, ge_range_all, df_train_output.columns)

    train_input_ext = np.repeat(df_train_input.as_matrix()[:, np.newaxis, :], 
                                nDepth, axis=1)

    keras_util.fitModel(model, train_input_ext, df_train_output.as_matrix(), [nDepth], alpha, "rnn", dir_path, fold_filename)

    test_input_ext = np.repeat(df_test_input.as_matrix()[:, np.newaxis, :], 
                               nDepth, axis=1)

    # Save test predictions:
    test_pred = model.predict(test_input_ext)
    df_test_pred = pd.DataFrame(data = test_pred,
                                columns=df_test_output.columns,
                                index=None)
    data_rw.savePreds(df_test_pred, dir_path, fold_filename, [nDepth], alpha, "rnn")
    
    # Save training predictions:
    train_pred = model.predict(train_input_ext)
    df_train_pred = pd.DataFrame(data = train_pred,
                                 columns=df_train_output.columns,
                                 index=None)
    data_rw.savePreds(df_train_pred, dir_path, fold_filename, [nDepth], alpha, "train_rnn")

dir_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ameen/mygithub/grnnPipeline/data/dream5/modules_gnw_a/size-20/top_edges-1_gnw_data/d_1/"
fold_filename = sys.argv[2] if len(sys.argv) > 2 else "n10_f1.txt"

data_filenames = {"NonTFs": '{!s}/processed_NonTFs.tsv'.format(dir_path),
                  "TFs" : '{!s}/processed_TFs.tsv'.format(dir_path),
                  "KOs" : '{!s}/processed_KO.tsv'.format(dir_path)}

ge_range_dic = data_rw.get_ge_range('{!s}/../ge_range.csv'.format(dir_path))

if len(sys.argv) <= 3:
    h_params = keras_util.getRnnHyperParameters(max_depth=5, min_alpha=0, max_alpha=5.0, num_combinations=50)
    for row in h_params:
        run(data_filenames, fold_filename, alpha=row[0], nDepth=row[1], ge_range_all=ge_range_dic)
else: #ToDo: should load config from a file instead
    run(data_filenames, fold_filename, alpha=0.0, nDepth=3, ge_range_all=ge_range_dic) #best architecture for app24_chemotaxis (based on d_1:d_10)

keras.backend.clear_session()
