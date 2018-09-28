import os
import sys
import csv
import numpy as np
import pandas as pd
import itertools as it
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l1
import lib.data_rw as data_rw
import lib.keras_util as keras_util

def getModel(nDInput, nDOutput, alpha):
    isFirst = True
    model = Sequential()
    model.add(Dense(nDOutput, input_shape=(nDInput,), kernel_regularizer=l1(alpha)))
    return model

def run(data_filenames, fold_filename, alpha):
    fold_filepath = '{!s}/folds/{!s}'.format(dir_path, fold_filename)
    layers_config = []
    df_train_input, df_train_output, df_test_input, df_test_output = data_rw.loadData(data_filenames, fold_filepath)
    model = getModel(nDInput = df_train_input.shape[1],
                     nDOutput = df_train_output.shape[1],
                     alpha = alpha)

    keras_util.fitModel(model, df_train_input.as_matrix(), df_train_output.as_matrix(), layers_config, alpha, "lasso", dir_path, fold_filename)
    
    # Save test predictions:
    test_pred = model.predict(df_test_input.as_matrix())
    df_test_pred = pd.DataFrame(data = test_pred,
                                columns=df_test_output.columns,
                                index=None)
    data_rw.savePreds(df_test_pred, dir_path, fold_filename, layers_config, alpha, "lasso")

    # Save training predictions:
    train_pred = model.predict(df_train_input.as_matrix())
    df_train_pred = pd.DataFrame(data = train_pred,
                                 columns=df_train_output.columns,
                                 index=None)
    data_rw.savePreds(df_train_pred, dir_path, fold_filename, layers_config, alpha, "train_lasso")

dir_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ameen/mygithub/GRNN_Clean/data/dream5_ecoli/expr1/modules_aggr/size-10/top_edges-1_gnw_data/d_1/"
fold_filename = sys.argv[2] if len(sys.argv) > 2 else "n10_f1.txt"

data_filenames = {"NonTFs": '{!s}/processed_NonTFs.tsv'.format(dir_path),
                  "TFs" : '{!s}/processed_TFs.tsv'.format(dir_path),
                  "KOs" : '{!s}/processed_KO.tsv'.format(dir_path)}

h_params = keras_util.getLassoHyperParameters(min_alpha=0, max_alpha=5, num_combinations=50)
for row in h_params:
    alpha = row[0]
    run(data_filenames, fold_filename, alpha)

keras.backend.clear_session()
